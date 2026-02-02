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
import cv2
import os
from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from fused_ssim import fused_ssim
from scene import GaussianModel, Camera
from utils.graphics_utils import (
    patch_offsets,
    patch_warp,
)
from gaussian_renderer import sample_depth, render
import warp_patch_ncc


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    return fused_ssim(img1, img2, padding="valid")


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def get_img_grad_weight(img):
    _, hd, wd = img.shape
    bottom_point = img[..., 2:hd, 1 : wd - 1]
    top_point = img[..., 0 : hd - 2, 1 : wd - 1]
    right_point = img[..., 1 : hd - 1, 2:wd]
    left_point = img[..., 1 : hd - 1, 0 : wd - 2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None, None], (1, 1, 1, 1), mode="constant", value=1.0).squeeze()
    return grad_img


def L1_loss_appearance(image, gt_image, gaussians, view_idx):
    app_model = gaussians.app_model
    if app_model is GaussianModel.App_model.NO:
        return l1_loss(image, gt_image)
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)

    if app_model is GaussianModel.App_model.GS:
        exposure = appearance_embedding
        transformed = torch.addmm(
            exposure[:3, 3, None],
            exposure[:3, :3],
            image.reshape(3, -1),
        ).reshape_as(image)
        return l1_loss(transformed, gt_image)

    if app_model is GaussianModel.App_model.GOF:
        origH, origW = image.shape[1:]
        H, W = origH // 32 * 32, origW // 32 * 32
        top, left = (origH - H) // 2, (origW - W) // 2

        crop = image[:, top : top + H, left : left + W]
        crop_gt = gt_image[:, top : top + H, left : left + W]

        down = torch.nn.functional.interpolate(crop[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]
        embedding_map = appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)
        net_in = torch.cat([down, embedding_map], dim=0)[None]

        mapping = gaussians.appearance_network(net_in)
        transformed = mapping * crop
        return l1_loss(transformed, crop_gt)

    if app_model is GaussianModel.App_model.PGSR:
        transformed = torch.addcmul(appearance_embedding[1], torch.exp(appearance_embedding[0]), image)
        return l1_loss(transformed, gt_image)


class PatchMatch:
    def __init__(self, patch_size, pixel_noise_th, kernel_size, pipe, debug=True, model_path=None):
        self.patch_size = patch_size
        self.total_patch_size = (patch_size * 2 + 1) ** 2
        self.pixel_noise_th = pixel_noise_th
        self.offsets = patch_offsets(patch_size, device="cuda") * 0.5
        self.offsets.requires_grad_(False)
        self.kernel_size = kernel_size
        self.pipe = pipe
        self.debug = debug
        self.model_path = model_path
        if debug:
            os.makedirs(os.path.join(model_path, "debug"), exist_ok=True)

    def __call__(self, gaussians: GaussianModel, render_pkg: dict, viewpoint_cam: Camera, nearest_cam: Camera, iteration=0, depth_normal=None):
        if nearest_cam is None:
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        ## compute geometry consistency mask
        with torch.no_grad():
            ix = (torch.arange(W, device="cuda", dtype=torch.float32) - viewpoint_cam.Cx) / viewpoint_cam.Fx
            iy = (torch.arange(H, device="cuda", dtype=torch.float32) - viewpoint_cam.Cy) / viewpoint_cam.Fy
            view_to_nearest_T = (
                -viewpoint_cam.world_view_transform[:3, :3].T @ nearest_cam.R @ nearest_cam.T + viewpoint_cam.world_view_transform[3, :3]
            )
            nearest_to_view_R = nearest_cam.R.transpose(1, 0) @ viewpoint_cam.world_view_transform[:3, :3]

        # pts = (rays_d * render_pkg["median_depth"].squeeze().unsqueeze(-1)).reshape(-1, 3)
        depth_reshape = render_pkg["median_depth"].squeeze().unsqueeze(-1)
        pts = torch.cat([depth_reshape * ix[None, :, None], depth_reshape * iy[:, None, None], depth_reshape], dim=-1)

        R = viewpoint_cam.R
        T = viewpoint_cam.T
        pts = (pts - T) @ R.T
        sampled_pkg = sample_depth(
            pts,
            nearest_cam,
            gaussians,
            self.pipe,
            self.kernel_size,
        )

        pts_in_nearest_cam = sampled_pkg["sampled_depth"]
        R = nearest_cam.R
        T = nearest_cam.T

        pts_in_view_cam = view_to_nearest_T + pts_in_nearest_cam @ nearest_to_view_R
        pts_projections = pts_in_view_cam[..., :2] / torch.clamp_min(pts_in_view_cam[..., 2:], 1e-7)
        pts_projections = torch.addcmul(
            pts_projections.new_tensor([viewpoint_cam.Cx, viewpoint_cam.Cy]),
            pts_projections.new_tensor([viewpoint_cam.Fx, viewpoint_cam.Fy]),
            pts_projections,
        )

        ix, iy = torch.meshgrid(
            torch.arange(W, device="cuda", dtype=torch.int32),
            torch.arange(H, device="cuda", dtype=torch.int32),
            indexing="xy",
        )
        pixels = torch.stack([ix, iy], dim=-1)
        pixel_f = pixels.type(torch.float32).requires_grad_(False)
        pixel_noise = torch.pairwise_distance(pts_projections, pixel_f)

        with torch.no_grad():
            d_mask = (
                sampled_pkg["inside"]
                & (pts_in_nearest_cam[..., -1] > 0.2)
                & (pts_in_view_cam[..., -1] > 0.2)
                & (pixel_noise < self.pixel_noise_th)
                & (render_pkg["median_depth"].squeeze() > 0)
            )
            weights = torch.exp(-pixel_noise)
            weights[~d_mask] = 0
        ##############################################

        if iteration % 200 == 0 and self.debug:
            with torch.no_grad():
                gt_img_show = (viewpoint_cam.original_image.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                img_show = ((render_pkg["render"]).permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                normal_show = (((render_pkg["normal"] + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
                if depth_normal is None:
                    depth_normal_show = (
                        (nearest_cam.original_image.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    )
                else:
                    depth_normal_show = (((depth_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
                d_mask_show = (weights.float() * 255).detach().cpu().numpy().astype(np.uint8)
                d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                depth = render_pkg["median_depth"].squeeze().detach().cpu().numpy()
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                row0 = np.concatenate([gt_img_show, img_show, depth_normal_show], axis=1)
                row1 = np.concatenate([d_mask_show_color, depth_color, normal_show], axis=1)
                image_to_show = np.concatenate([row0, row1], axis=0)
                cv2.imwrite(os.path.join(self.model_path, "debug", "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
        ################## Compute NCC for warped patches ##################
        if not d_mask.any():
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")

        geo_loss = ((weights * pixel_noise)[d_mask]).mean()
        with torch.no_grad():
            d_mask = torch.flatten(d_mask)
            valid_indices = torch.argwhere(d_mask).squeeze(1)
            weights = torch.flatten(weights)[valid_indices]
            pixels = torch.index_select(pixels.view(-1, 2), dim=0, index=valid_indices)
            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

        depth_select = torch.index_select(render_pkg["median_depth"].view(-1), dim=0, index=valid_indices)
        normal_select = torch.index_select(render_pkg["normal"].view(3, -1), dim=1, index=valid_indices).transpose(1, 0)
        normal_select = F.normalize(normal_select, dim=-1)

        cc, valid_mask = warp_patch_ncc.warp_patch_ncc(
            depth_select,
            normal_select,
            pixels,
            ref_to_neareast_r.T,
            ref_to_neareast_t,
            viewpoint_cam.gray_image.to("cuda").squeeze(),
            nearest_cam.gray_image.to("cuda").squeeze(),
            viewpoint_cam.Fx,
            viewpoint_cam.Fy,
            viewpoint_cam.Cx,
            viewpoint_cam.Cy,
            nearest_cam.Fx,
            nearest_cam.Fy,
            nearest_cam.Cx,
            nearest_cam.Cy,
            False,
        )
        ncc = torch.clamp(1 - cc, 0.0, 2.0)
        ncc_mask = (ncc < 0.9) & valid_mask

        ncc = ncc.squeeze() * weights
        ncc = ncc[ncc_mask.squeeze()]

        if ncc_mask.any():
            ncc_loss = ncc.mean()
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        return ncc_loss, geo_loss
