import os
import numpy as np
import cv2
import pickle
import json

import torch
from torch.utils.data import Dataset
from scipy.spatial import KDTree

from data.utils import load_yaml, set_seed, fps_rad, fps_np, recenter, opengl2cam, depth2fgpcd, pcd2pix
from config import gen_args


class ParticleDataset(Dataset):
    def __init__(self, data_dir, args, phase, cam):
        self.args = args

        n_episode = 2000
        n_timestep = 10
        self.global_scale = 24

        train_valid_ratio = 0.9

        n_train = int(n_episode * train_valid_ratio)
        n_valid = n_episode - n_train

        if phase == 'train':
            self.epi_st_idx = 0
            self.n_episode = n_train
        elif phase == 'valid':
            self.epi_st_idx = n_train
            self.n_episode = n_valid

        self.n_timestep = n_timestep + 1
        self.n_his = 1
        self.n_roll = 5
        self.data_dir = data_dir

        self.screenHeight = 720
        self.screenWidth = 720
        self.img_channel = 1

        self.cam_params, self.cam_extrinsic = cam

    def __len__(self):
        return self.n_episode * (self.n_timestep - self.n_his - self.n_roll + 1)
    
    def read_particles(self, particles_path):
        particles = np.load(particles_path).reshape(-1, 4)
        particles[:, 3] = 1.0
        opencv_T_opengl = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        opencv_T_world = np.matmul(np.linalg.inv(self.cam_extrinsic), opencv_T_opengl)
        particles = np.matmul(np.linalg.inv(opencv_T_world), particles.T).T[:, :3] / self.global_scale
        return particles

    def get(self, idx):
        particle_den_min = 15
        particle_den_max = 6500
        particle_den = np.random.uniform(particle_den_min, particle_den_max)
        particle_r = 1 / np.sqrt(particle_den)

        offset = self.n_timestep - self.n_his - self.n_roll + 1
        idx_episode = idx // offset + self.epi_st_idx
        idx_timestep = idx % offset
        print('idx_episode', idx_episode, 'idx_timestep', idx_timestep)

        action_path = os.path.join(self.data_dir, '%d/actions.p' % idx_episode)
        with open(action_path, 'rb') as fp:
            actions = pickle.load(fp)

        # sample particles to track in the first frame
        first_depth_path = os.path.join(self.data_dir, '%d/%d_depth.png' % (idx_episode, idx_timestep))
        first_depth = cv2.imread(first_depth_path, cv2.IMREAD_ANYDEPTH) / (self.global_scale * 1000.0)
        first_depth_fgpcd = depth2fgpcd(first_depth, (first_depth < 0.599/0.8), self.cam_params) # [N, 3]
        sampled_pts = fps_rad(first_depth_fgpcd, particle_r) # [particle_num, 3]
        particle_num = sampled_pts.shape[0]
        sampled_pts = recenter(first_depth_fgpcd, sampled_pts, r = min(0.02, 0.5 * particle_r)) # [particle_num, 3]

        # find the nearest gt particle to sampled_pts
        first_particles_path = os.path.join(self.data_dir, '%d/%d_particles.npy' % (idx_episode, idx_timestep))
        first_particles = self.read_particles(first_particles_path)
        
        first_particles_tree = KDTree(first_particles)
        _, nearest_idx = first_particles_tree.query(sampled_pts, k=1)
        
        states = np.zeros((self.n_his + self.n_roll, particle_num, 3))
        color_imgs = np.zeros((self.n_his + self.n_roll, 720, 720, 3)).astype(np.uint8)
        states_delta = np.zeros((self.n_his + self.n_roll - 1, particle_num, 3))
        attrs = np.zeros(states.shape[:2])

        for i in range(idx_timestep, idx_timestep + self.n_his + self.n_roll):
            particles_path = os.path.join(self.data_dir, '%d/%d_particles.npy' % (idx_episode, i))
            particles = self.read_particles(particles_path)
            states[i - idx_timestep] = particles[nearest_idx, :]

            if i < idx_timestep + self.n_his + self.n_roll - 1:
                s = actions[i, :2]
                e = actions[i, 2:]
                h = 0.0
                pusher_w = 0.8 / 24.0

                s_3d = np.array([s[0], h, -s[1]])
                e_3d = np.array([e[0], h, -e[1]])
                s_3d_cam = opengl2cam(s_3d[None, :], self.cam_extrinsic, self.global_scale)[0]
                e_3d_cam = opengl2cam(e_3d[None, :], self.cam_extrinsic, self.global_scale)[0]
                push_dir_cam = e_3d_cam - s_3d_cam
                push_l = np.linalg.norm(push_dir_cam)
                push_dir_cam = push_dir_cam / push_l
                assert abs(push_dir_cam[2]) < 1e-6

                push_dir_ortho_cam = np.array([-push_dir_cam[1], push_dir_cam[0], 0.0])
                pos_diff_cam = particles[nearest_idx, :] - s_3d_cam[None, :] # [particle_num, 3]
                pos_diff_ortho_proj_cam = (pos_diff_cam * np.tile(push_dir_ortho_cam[None, :], (particle_num, 1))).sum(axis=1) # [particle_num,]
                pos_diff_proj_cam = (pos_diff_cam * np.tile(push_dir_cam[None, :], (particle_num, 1))).sum(axis=1) # [particle_num,]
                pos_diff_l_mask = ((pos_diff_proj_cam < push_l) & (pos_diff_proj_cam > 0.0)).astype(np.float32) # hard mask
                pos_diff_w_mask = np.maximum(np.maximum(-pusher_w - pos_diff_ortho_proj_cam, 0.), # soft mask
                                            np.maximum(pos_diff_ortho_proj_cam - pusher_w, 0.))
                pos_diff_w_mask = np.exp(-pos_diff_w_mask / 0.01) # [particle_num,]
                pos_diff_to_end_cam = (e_3d_cam[None, :] - particles[nearest_idx, :]) # [particle_num, 3]
                pos_diff_to_end_cam = (pos_diff_to_end_cam * np.tile(push_dir_cam[None, :], (particle_num, 1))).sum(axis=1) # [particle_num,]
                states_delta[i - idx_timestep] = pos_diff_to_end_cam[:, None] * push_dir_cam[None, :] * pos_diff_l_mask[:, None] * pos_diff_w_mask[:, None]
            color_img_path = os.path.join(self.data_dir, '%d/%d_color.png' % (idx_episode, i))
            color_img = cv2.imread(color_img_path)
            color_imgs[i - idx_timestep, :, :, :] = color_img
        states = torch.FloatTensor(states)
        states_delta = torch.FloatTensor(states_delta)
        attrs = torch.FloatTensor(attrs)
        return states, states_delta, attrs, particle_num, particle_den, color_imgs


def dataset_test():
    args = gen_args()

    import pyflex

    screenWidth = 720
    screenHeight = 720
    headless = True
    pyflex.set_screenWidth(screenWidth)
    pyflex.set_screenHeight(screenHeight)
    pyflex.set_light_dir(np.array([0.1, 2.0, 0.1]))
    pyflex.set_light_fov(70.)
    pyflex.init(headless)

    cam_idx = 0
    rad = np.deg2rad(cam_idx * 20.)
    global_scale = 24
    cam_dis = 0.0 * global_scale / 8.0
    cam_height = 6.0 * global_scale / 8.0
    camPos = np.array([np.sin(rad) * cam_dis, cam_height, np.cos(rad) * cam_dis])
    camAngle = np.array([rad, -np.deg2rad(90.), 0.])

    pyflex.set_camPos(camPos)
    pyflex.set_camAngle(camAngle)

    projMat = pyflex.get_projMatrix().reshape(4, 4).T
    cx = screenWidth / 2.0
    cy = screenHeight / 2.0
    fx = projMat[0, 0] * cx
    fy = projMat[1, 1] * cy
    cam_params =  [fx, fy, cx, cy]

    cam_extrinsics = np.array(pyflex.get_viewMatrix()).reshape(4, 4).T
    cam = [cam_params, cam_extrinsics]

    dataset = ParticleDataset(
        data_dir="/home/zhangkaifeng/projects/dyn-res-pile-manip/data/gnn_dyn_data", 
        args=args,
        phase='train',
        cam=cam
    )
    states, states_delta, attrs, particle_num, particle_den, color_imgs = dataset.get(0)

    vid = cv2.VideoWriter('dataset_nopusher.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (720, 720))
    for i in range(states.shape[0] - 1):
        img = color_imgs[i]
        obj_pix = pcd2pix(states[i], cam[0])
        next_pix = pcd2pix(states[i] + states_delta[i], cam[0])
        for j in range(obj_pix.shape[0]):
            img = cv2.circle(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])), 5, (0, 0, 255), -1)
        for j in range(next_pix.shape[0]):
            img = cv2.arrowedLine(img.copy(), (int(obj_pix[j, 1]), int(obj_pix[j, 0])),
                                  (int(next_pix[j, 1]), int(next_pix[j, 0])), (0, 255, 0), 2)
        vid.write(img)
    vid.release()


if __name__ == '__main__':
    dataset_test()
