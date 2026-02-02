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

import math
from typing import NamedTuple

import cv2
import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """get world 2 camera matrix

    Args:
        R (_type_): c2w rotation
        t (_type_): w2c camera center
        translate (_type_, optional): _description_. Defaults to np.array([.0, .0, .0]).
        scale (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    # compose w2c matrix
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # invert to get c2w
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # get the final w2c matrix
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def depth_to_normal(view, depth):
    W, H = view.image_width, view.image_height
    x = (torch.arange(W, device="cuda", dtype=torch.float32) - view.Cx) / view.Fx
    y = (torch.arange(H, device="cuda", dtype=torch.float32) - view.Cy) / view.Fy
    points = torch.cat([depth * x[None, None], depth * y[None, :, None], depth], dim=0)
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=0), dim=0)
    output = torch.nn.functional.pad(normal_map, (1, 1, 1, 1))

    valid_depths = depth > 0
    valid_depths = (
        valid_depths[:, 2:, 1:-1] & valid_depths[:, :-2, 1:-1] & valid_depths[:, 1:-1, 2:] & valid_depths[:, 1:-1, :-2] & valid_depths[:, 1:-1, 1:-1]
    )
    valid_points = torch.zeros_like(depth, dtype=torch.bool)
    valid_points[:, 1:-1, 1:-1] = valid_depths
    return output, valid_points


def bilinear_sampler(img, coords, mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


# project the reference point cloud into the source view, then project back
# extrinsics here refers c2w
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]),
    )
    # source 3D space
    xyz_src = np.matmul(
        np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
        np.vstack((xyz_ref, np.ones_like(x_ref))),
    )[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]),
    )
    # reference 3D space
    xyz_reprojected = np.matmul(
        np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
        np.vstack((xyz_src, np.ones_like(x_ref))),
    )[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src,
    thre1=0.5,
    thre2=0.01,
):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src,
    )
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < thre1, relative_depth_diff < thre2)
    # mask = dist < 0.2
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff


def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device, dtype=torch.float32)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing="xy")[::-1], dim=-1).view(1, -1, 2)


def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B, P, 1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid


def get_points_depth_in_depth_map(fov_camera, depth, points_in_camera_space, scale=1):
    st = max(int(scale / 2) - 1, 0)
    depth_view = depth[None, :, st::scale, st::scale]
    W, H = int(fov_camera.image_width / scale), int(fov_camera.image_height / scale)
    depth_view = depth_view[..., :H, :W]
    pts_projections = (
        torch.stack(
            [
                points_in_camera_space[:, 0] * fov_camera.Fx / points_in_camera_space[:, 2] + fov_camera.Cx,
                points_in_camera_space[:, 1] * fov_camera.Fy / points_in_camera_space[:, 2] + fov_camera.Cy,
            ],
            -1,
        ).float()
        / scale
    )
    mask = (
        (pts_projections[:, 0] > 0)
        & (pts_projections[:, 0] < W)
        & (pts_projections[:, 1] > 0)
        & (pts_projections[:, 1] < H)
        & (points_in_camera_space[:, 2] > 0.1)
    )

    pts_projections[..., 0] /= (W - 1) / 2
    pts_projections[..., 1] /= (H - 1) / 2
    pts_projections -= 1
    pts_projections = pts_projections.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, :, :, 0]
    return map_z, mask


def get_points_from_depth(fov_camera, depth, scale=1):
    st = int(max(int(scale / 2) - 1, 0))
    depth_view = depth.squeeze()[st::scale, st::scale]
    W, H = fov_camera.image_width, fov_camera.image_height
    ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    rays_d = (
        torch.stack(
            [
                (ix - fov_camera.Cx) / fov_camera.Fx * scale,
                (iy - fov_camera.Cy) / fov_camera.Fy * scale,
                torch.ones_like(ix),
            ],
            -1,
        )
        .float()
        .cuda()
    )
    depth_view = depth_view[: rays_d.shape[0], : rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1, 3)
    R = fov_camera.R
    T = fov_camera.T
    pts = (pts - T) @ R.transpose(1, 0)
    return pts
