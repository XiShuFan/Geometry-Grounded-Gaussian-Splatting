#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
from typing import Any, Sequence, Union

import numpy as np
import torch

from arguments import ModelParams
from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        print(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.masks, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Loading Training Cameras: {len(self.train_cameras[resolution_scale])} .")

            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"Loading Test Cameras: {len(self.test_cameras[resolution_scale])} .")

            print("computing nearest_id")
            camera_centers_list: list[torch.Tensor] = []
            center_rays_list: list[torch.Tensor] = []
            with torch.no_grad():
                for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                    camera_centers_list.append(cur_cam.camera_center)
                    R = cur_cam.R
                    center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
                    center_ray = center_ray @ R.transpose(-1, -2)
                    center_rays_list.append(center_ray)
                camera_centers = torch.stack(camera_centers_list, dim=0)
                center_rays = torch.stack(center_rays_list, dim=0)
                center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
                diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
                tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1)
                angles_torch = torch.arccos(tmp) * 180 / 3.14159
                angles_np = angles_torch.detach().cpu().numpy()
                with open(os.path.join(self.model_path, "multi_view.json"), "w") as file:
                    for id, cur_cam in enumerate(self.train_cameras[resolution_scale]):
                        sorted_indices = np.lexsort((angles_np[id], diss[id]))
                        # sorted_indices = np.lexsort((diss[id], angles[id]))
                        mask = (
                            (angles_np[id][sorted_indices] < args.multi_view_max_angle)
                            & (diss[id][sorted_indices] > args.multi_view_min_dis)
                            & (diss[id][sorted_indices] < args.multi_view_max_dis)
                        )
                        sorted_indices = sorted_indices[mask]
                        multi_view_num = min(args.multi_view_num, len(sorted_indices))
                        json_d = {"ref_name": cur_cam.image_name, "nearest_name": []}
                        for index in sorted_indices[:multi_view_num]:
                            cur_cam.nearest_id.append(index)
                            # cur_cam.nearest_names.append(self.train_cameras[resolution_scale][index].image_name)
                            json_d["nearest_name"].append(self.train_cameras[resolution_scale][index].image_name)
                        json_str = json.dumps(json_d, separators=(",", ":"))
                        file.write(json_str)
                        file.write("\n")

        self.gaussians.create_app_model(len(scene_info.train_cameras), args.use_decoupled_appearance)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            with torch.no_grad():
                for camera_center in camera_centers_list:
                    dists_cam_gauss = torch.norm(self.gaussians.get_xyz - camera_center[None, :], dim=1)
                    max_scale = 0.05 * dists_cam_gauss.flatten()
                    log_max_scale = torch.log(max_scale).repeat(3, 1).permute(1, 0)
                    self.gaussians._scaling[:] = torch.clamp_max(self.gaussians._scaling, log_max_scale)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0) -> Sequence[Camera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> Sequence[Camera]:
        return self.test_cameras[scale]
