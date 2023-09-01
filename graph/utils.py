import numpy as np
import torch
import open3d as o3d
from dgl.geometry import farthest_point_sampler


def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def downsample_pcd(pcd, voxel_size):
    # pcd: (n, 3) numpy array
    # downpcd: (m, 3)
    
    # convert numpy array to open3d point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

    downpcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    downpcd = np.asarray(downpcd_o3d.points)


def fps(pcd, particle_num, init_idx=-1):
    # pcd: (n, 3) numpy array
    # pcd_fps: (self.particle_num, 3) numpy array
    pcd_tensor = torch.from_numpy(pcd).float()[None, ...]
    if init_idx == -1:
        # init_idx = findClosestPoint(pcd, pcd.mean(axis=0))
        pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num)[0]
    else:
        pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num, init_idx)[0]
    pcd_fps_tensor = pcd_tensor[0, pcd_fps_idx_tensor]
    pcd_fps = pcd_fps_tensor.numpy()
    dist = np.linalg.norm(pcd[:, None] - pcd_fps[None, :], axis=-1)
    dist = dist.min(axis=1)
    return pcd_fps, dist.max()  # max_i (min_j ||pcd[i] - pcd_fps[j]||)


def recenter(pcd, sampled_pcd, r = 0.02):
    # pcd: (n, 3) numpy array
    # sampled_pcd: (self.partcile_num, 3) numpy array
    # recentering around a local point cloud
    particle_num = sampled_pcd.shape[0]
    dist = np.linalg.norm(pcd[:, None, :] - sampled_pcd[None, :, :], axis=2) # (n, self.particle_num)
    recenter_sampled_pcd = np.zeros_like(sampled_pcd)
    for i in range(particle_num):
        recenter_sampled_pcd[i] = pcd[dist[:, i] < r].mean(axis=0)
    return recenter_sampled_pcd

