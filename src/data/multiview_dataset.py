
import os
import numpy as np
import cv2
import torch
from PIL import Image

from data.utils import load_yaml, set_seed, fps, fps_rad, recenter, \
        opengl2cam, depth2fgpcd, pcd2pix, find_relations_neighbor

class MultiviewParticleDataset:
    def __init__(self, args, depths=None, masks=None, cams=None, 
            text_labels_list=None, material_dict=None):
        self.args = args
        self.depths = depths  # list of PIL depth images
        self.masks = masks  # list of masks
        self.cam_params, self.cam_extrinsics = cams  # (4,) * n_cameras, (4, 4) * n_cameras
        self.text_labels = text_labels_list  # list of text labels
        self.material_dict = material_dict  # dict of {obj_name: material_name}

        self.global_scale = 24
        self.depth_thres = 0.599 / 0.8
        self.particle_num = 50
        self.adj_thresh = 0.1

        self.n_cameras = len(self.depths)

        for camera_index in range(self.n_cameras):
            depth = np.array(self.depths[camera_index])
            masks = np.array(self.masks[camera_index])
            cam_param = self.cam_params[camera_index]
            cam_extrinsic = self.cam_extrinsics[camera_index]
            pcd_list = self.parse_pcd(depth, masks, cam_param, cam_extrinsic)
            import ipdb; ipdb.set_trace()

        pcd_label_list = np.unique(self.pcd_label)
        self.n_instance = pcd_label_list.shape[0]
        pcd_sample_list = []
        # particle_num_list = []
        particle_r_list = []
        particle_den_list = []
        for i in range(pcd_label_list.shape[0]):
            # particle_num_i = int((self.pcd_label == i).sum())
            pcd_i = self.pcd[self.pcd_label[:, 0] == i]
            pcd_sample_i, _, particle_r_i, particle_den_i = self.subsample(pcd_i, particle_num=self.particle_num)
            pcd_sample_list.append(pcd_sample_i)
            # particle_num_list.append(particle_num_i)
            particle_r_list.append(particle_r_i)
            particle_den_list.append(particle_den_i)

        # TODO check if taking mean is good
        self.particle_r = np.array(particle_r_list).mean()
        self.particle_den = np.array(particle_den_list).mean()

        pcd_sample = np.vstack(pcd_sample_list)

        self.attrs = np.zeros((self.particle_num * self.n_instance, args.attr_dim), dtype=pcd_sample.dtype) # [particle_num, attr_dim]
        self.state = pcd_sample
        self.action = None
        self.Rr = None
        self.Rs = None
        self.history_states = []

    def update(self, state):
        self.history_states.append(self.state.copy())
        self.state = state

    def parse_pcd(self, depth, masks, cam_param, cam_extrinsic):
        pcd_list = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = np.logical_and(mask, depth > 0)

            # to camera frame
            fgpcd = np.zeros((mask.sum(), 3))
            fgpcd_label = np.zeros((mask.sum(), 2))
            fx, fy, cx, cy = cam_param
            pos_x, pos_y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))  # w, h
            pos_x = pos_x[mask]
            pos_y = pos_y[mask]
            fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
            fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
            fgpcd[:, 2] = depth[mask]

            # to world frame
            fgpcd = np.hstack((fgpcd, np.ones((fgpcd.shape[0], 1))))
            fgpcd = np.matmul(fgpcd, np.linalg.inv(cam_extrinsic).T)[:, :3]

            pcd_list.append(fgpcd)
        return pcd_list

    def get_grouping(self):
        n_instance = np.unique(self.pcd_label).shape[0]
        n_p = self.particle_num * n_instance
        p_instance = torch.ones((1, n_p, n_instance), dtype=torch.float32) # the group each particle belongs to
        p_rigid = torch.zeros((1, n_instance), dtype=torch.float32) # the rigidness of each group

        # for i in range(n_instance):
        #     # import ipdb; ipdb.set_trace()
        #     p_instance[0, :, i] = torch.tensor((self.pcd_label[:, 0] == i).astype(np.float32))

        return n_p, n_instance, p_instance, p_rigid

    def subsample(self, pcd, particle_num=None, particle_r=None, particle_den=None):
        if particle_num is not None:
            sampled_pts, particle_r = fps(pcd, particle_num)
            particle_den = 1 / particle_r ** 2
        elif particle_r is not None:
            sampled_pts = fps_rad(pcd, particle_r) # [particle_num, 3]
            particle_den = 1 / particle_r ** 2
        elif particle_den is not None:
            particle_r = 1 / np.sqrt(particle_den)
            sampled_pts = fps_rad(pcd, particle_r) # [particle_num, 3]
        else:
            raise AssertionError("No subsampling method specified")
        particle_num = sampled_pts.shape[0]
        sampled_pts = recenter(pcd, sampled_pts, r = min(0.02, 0.5 * particle_r)) # [particle_num, 3]
        return sampled_pts, particle_num, particle_r, particle_den

    def generate_relation(self):
        args = self.args
        B = 1
        N = self.particle_num * self.n_instance
        rels = []

        s_cur = torch.tensor(self.state).unsqueeze(0)
        s_delta = torch.tensor(self.action).unsqueeze(0)

        # s_receiv, s_sender: B x particle_num x particle_num x 3
        s_receiv = (s_cur + s_delta)[:, :, None, :].repeat(1, 1, N, 1)
        s_sender = (s_cur + s_delta)[:, None, :, :].repeat(1, N, 1, 1)

        # dis: B x particle_num x particle_num
        # adj_matrix: B x particle_num x particle_num
        threshold = self.adj_thresh * self.adj_thresh
        dis = torch.sum((s_sender - s_receiv)**2, -1)
        max_rel = min(10, N)
        topk_res = torch.topk(dis, k=max_rel, dim=2, largest=False)
        topk_idx = topk_res.indices
        topk_bin_mat = torch.zeros_like(dis, dtype=torch.float32)
        topk_bin_mat.scatter_(2, topk_idx, 1)
        adj_matrix = ((dis - threshold) < 0).float()
        adj_matrix = adj_matrix * topk_bin_mat

        n_rels = adj_matrix.sum(dim=(1,2))
        n_rel = n_rels.max().long().item()
        rels_idx = []
        rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
        rels_idx = torch.hstack(rels_idx).to(dtype=torch.long)
        rels = adj_matrix.nonzero()
        Rr = torch.zeros((B, n_rel, N), dtype=s_cur.dtype)
        Rs = torch.zeros((B, n_rel, N), dtype=s_cur.dtype)
        Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1  # batch_idx, rel_idx, receiver_particle_idx
        Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1  # batch_idx, rel_idx, sender_particle_idx

        self.Rr = Rr.squeeze(0).numpy() # n_rel, n_particle
        self.Rs = Rs.squeeze(0).numpy() # n_rel, n_particle
        return self.Rr, self.Rs

    def parse_action(self, action):
        # action: [start_x, start_y, end_x, end_y]
        s = action[:2]
        e = action[2:]
        h = 0.0
        pusher_w = 0.8 / 24.0
        s_3d = np.array([s[0], h, -s[1]])
        e_3d = np.array([e[0], h, -e[1]])
        s_3d_cam = opengl2cam(s_3d[None, :], self.cam_extrinsics, self.global_scale)[0]
        e_3d_cam = opengl2cam(e_3d[None, :], self.cam_extrinsics, self.global_scale)[0]
        push_dir_cam = e_3d_cam - s_3d_cam
        push_l = np.linalg.norm(push_dir_cam)
        push_dir_cam = push_dir_cam / push_l
        assert abs(push_dir_cam[2]) < 1e-6

        push_dir_ortho_cam = np.array([-push_dir_cam[1], push_dir_cam[0], 0.0])
        pos_diff_cam = self.state - s_3d_cam[None, :] # [particle_num, 3]
        pos_diff_ortho_proj_cam = (pos_diff_cam * np.tile(push_dir_ortho_cam[None, :], (self.particle_num * self.n_instance, 1))).sum(axis=1) # [particle_num,]
        pos_diff_proj_cam = (pos_diff_cam * np.tile(push_dir_cam[None, :], (self.particle_num * self.n_instance, 1))).sum(axis=1) # [particle_num,]
        pos_diff_l_mask = ((pos_diff_proj_cam < push_l) & (pos_diff_proj_cam > 0.0)).astype(np.float32) # hard mask
        pos_diff_w_mask = np.maximum(np.maximum(-pusher_w - pos_diff_ortho_proj_cam, 0.), # soft mask
                                    np.maximum(pos_diff_ortho_proj_cam - pusher_w, 0.))
        pos_diff_w_mask = np.exp(-pos_diff_w_mask / 0.01) # [particle_num,]
        pos_diff_to_end_cam = (e_3d_cam[None, :] - self.state) # [particle_num, 3]
        pos_diff_to_end_cam = (pos_diff_to_end_cam * np.tile(push_dir_cam[None, :], (self.particle_num * self.n_instance, 1))).sum(axis=1) # [particle_num,]
        states_delta = pos_diff_to_end_cam[:, None] * push_dir_cam[None, :] * pos_diff_l_mask[:, None] * pos_diff_w_mask[:, None]

        self.action = states_delta
        return states_delta
