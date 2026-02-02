import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def extract_mesh(dataset, pipe, iteration, num_cluster=1):
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    kernel_size = dataset.kernel_size

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = scene.getTrainCameras()

    depth_list = []
    color_list = []
    for viewpoint_cam in viewpoint_cam_list:
        # Rendering offscreen from that camera
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1, 2, 0)
        color_list.append(np.ascontiguousarray(rendered_img))
        depth = render_pkg["median_depth"].clone()
        if viewpoint_cam.gt_mask is not None:
            depth[(viewpoint_cam.gt_mask < 0.5)] = 0
        depth_list.append(depth[0].cpu().numpy())

    torch.cuda.empty_cache()
    voxel_size = 0.002
    o3d_device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=o3d_device,
    )
    for color, depth, viewpoint_cam in zip(color_list, depth_list, viewpoint_cam_list):
        depth = o3d.t.geometry.Image(depth)
        depth = depth.to(o3d_device)
        color = o3d.t.geometry.Image(color)
        color = color.to(o3d_device)
        intrinsic = np.array([[viewpoint_cam.Fx, 0, viewpoint_cam.Cx], [0, viewpoint_cam.Fy, viewpoint_cam.Cy], [0, 0, 1]], dtype=np.float64)
        intrinsic = o3d.core.Tensor(intrinsic)
        extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1.0, 8.0)
        vbg.integrate(frustum_block_coords, depth, color, intrinsic, extrinsic, 1.0, 8.0)

    mesh = vbg.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, "recon.ply"), mesh.to_legacy())
    mesh = post_process_mesh(mesh.to_legacy(), num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, "recon_post.ply"), mesh)
    print("done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    with torch.no_grad():
        extract_mesh(model.extract(args), pipeline.extract(args), args.iteration, args.num_cluster)
