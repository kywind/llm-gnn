import os
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl
import json
import open3d as o3d
from dgl.geometry import farthest_point_sampler

from config import gen_args
from data.utils import label_colormap, opengl2cam
from preprocess import construct_edges_from_states
from gnn.model_wrapper import gen_model


def extract_push_sub(data_dir, episode_idx, start_frame, end_frame):
    obj_ptcl_start = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{start_frame}_particles.npy"))
    obj_ptcl_end = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{end_frame}_particles.npy"))
    obj_ptcl_start = obj_ptcl_start[:, :3]
    obj_ptcl_end = obj_ptcl_end[:, :3]
    obj_kp = np.stack([obj_ptcl_start, obj_ptcl_end], axis=0)

    y = 0.5
    eef_ptcl_start = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{start_frame}_endeffector.npy"))
    eef_ptcl_end = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{end_frame}_endeffector.npy"))
    x_start = eef_ptcl_start[0]
    z_start = eef_ptcl_start[1]
    x_end = eef_ptcl_end[0]
    z_end = eef_ptcl_end[1]
    pos_start = np.array([[x_start, y, z_start]])  # (1, 3)
    pos_end = np.array([[x_end, y, z_end]])
    eef_kp = np.stack([pos_start, pos_end], axis=0)  # (2, 1, 3)
    eef_kp[:, :, 2] *= -1
    return obj_kp, eef_kp


def extract_pushes(args, data_dir, save_dir, dist_thresh_range=[0.05, 0.10]):
    # use overlapping samples
    # provide canonical frame info
    # compatible to other data layouts (make a general episode list)

    # dense keypoints
    # eef_kypts_dir = os.path.join(save_dir, 'kypts', 'eef_kypts') # list of (2, eef_ptcl_num, 3) for each push, indexed by push_num (dense)
    # os.makedirs(eef_kypts_dir, exist_ok=True)
    # obj_kypts_dir = os.path.join(save_dir, 'kypts', 'obj_kypts') # list of (ptcl_num, 3) for each push, indexed by push_num (dense)
    # os.makedirs(obj_kypts_dir, exist_ok=True)
    # physics_dir = os.path.join(save_dir, 'kypts', 'physics')
    # os.makedirs(physics_dir, exist_ok=True)
    frame_idx_dir = os.path.join(save_dir, 'frame_pairs')
    os.makedirs(frame_idx_dir, exist_ok=True)
    can_dir = os.path.join(save_dir, 'canonical_pos')
    os.makedirs(can_dir, exist_ok=True)

    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))

    print(f"Preprocessing starts. num_episodes: {num_episodes}")

    for episode_idx in range(num_episodes):
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_color.jpg"))))
        print(f"Processing episode {episode_idx}, num_frames: {num_frames}")

        actions = np.load(os.path.join(data_dir, f"episode_{episode_idx}/actions.npy"))
        steps = np.load(os.path.join(data_dir, f"episode_{episode_idx}/steps.npy"))
        steps_a = np.concatenate([[2], steps], axis=0)

        if len(actions) != len(steps):
            import ipdb; ipdb.set_trace()
            continue

        frame_idxs = []

        # get start-end pairs
        cnt = 0
        for fj in range(num_frames):

            # get canonical particles
            if fj == 0:
                particles = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{fj}_particles.npy"))[:, :3]
                # find main axis using PCA
                n = particles.shape[0]
                particles_mean = np.mean(particles, axis=0)
                particles_centered = particles - particles_mean
                cov = np.matmul(particles_centered.T, particles_centered) / n
                eig_val, eig_vec = np.linalg.eig(cov)
                eig_vec = eig_vec[:, np.argmax(eig_val)]
                eig_vec = eig_vec / np.linalg.norm(eig_vec)
                # save the particle positions along the main axis
                particles_can = (particles_centered * eig_vec).sum(1)
                np.save(os.path.join(can_dir, f'{episode_idx}.npy'), particles_can)

            curr_step = None
            for si in range(len(steps_a) - 1):
                # if step is [x1, x2]: [2, x1-2] is first push, (x1-1 is the final state of first push), 
                # [x1, x2-2] is second push
                if fj >= steps_a[si] and fj <= steps_a[si+1] - 2:
                    curr_step = si
                    break
            else:
                continue  # this frame is not valid
            assert curr_step is not None

            curr_frame = fj

            start_frame = steps_a[curr_step]
            end_frame = steps_a[curr_step + 1] - 2  # inclusive

            # action = actions[curr_step]
            # x_start, z_start, x_end, z_end = action
            # dist = np.sqrt((x_start - x_end) ** 2 + (z_start - z_end) ** 2)

            eef_particles_curr = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{curr_frame}_endeffector.npy"))

            fi = fj
            while fi <= end_frame:
                eef_particles_fi = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/{fi}_endeffector.npy"))
                x_curr = eef_particles_curr[0]
                z_curr = eef_particles_curr[1]
                x_fi = eef_particles_fi[0]
                z_fi = eef_particles_fi[1]
                dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)

                if dist_curr >= dist_thresh_range[0] and dist_curr < dist_thresh_range[1]:
                    cnt += 1
                    # obj_kp, eef_kp = extract_push_sub(data_dir, episode_idx, curr_frame, fi)
                    # save_path = os.path.join(obj_kypts_dir, f'{episode_idx:06}_{curr_frame:06}_{fi:06}.npy')
                    # np.save(save_path, obj_kp)
                    # save_path = os.path.join(eef_kypts_dir, f'{episode_idx:06}_{curr_frame:06}_{fi:06}.npy')
                    # np.save(save_path, eef_kp)
                    # prop_np = np.array([
                    #     properties['particle_radius'] * 10,
                    #     properties['num_particles'] * 0.001,
                    #     properties['length'] * 0.05,
                    #     properties['thickness'] * 0.02,
                    #     properties['dynamic_friction'] * 2,
                    #     properties['cluster_spacing'] * 0.1,
                    #     properties['global_stiffness'] * 1,
                    # ])  # approximate normalization
                    # save_path = os.path.join(physics_dir, f'{episode_idx:06}_{curr_frame:06}_{fi:06}.npy')
                    # np.save(save_path, prop_np)

                    frame_idxs.append([curr_frame, fi])

                    img_vis = False
                    if img_vis:
                        cam_idx = 0
                        obj_kp, eef_kp = extract_push_sub(data_dir, episode_idx, curr_frame, fi)
                        intr = np.load(os.path.join(data_dir, f"camera_intrinsic_params.npy"))[cam_idx]
                        extr = np.load(os.path.join(data_dir, f"camera_extrinsic_matrix.npy"))[cam_idx]
                        img = cv2.imread(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{curr_frame}_color.jpg"))
                        # transform particle to camera coordinate
                        particle_cam = opengl2cam(obj_kp[0], extr)
                        assert particle_cam[:, 2].min() > 0  # z > 0
                        fx, fy, cx, cy = intr
                        particle_projs = np.zeros((particle_cam.shape[0], 2))
                        particle_projs[:, 0] = particle_cam[:, 0] * fx / particle_cam[:, 2] + cx
                        particle_projs[:, 1] = particle_cam[:, 1] * fy / particle_cam[:, 2] + cy
                        for pi in range(particle_projs.shape[0]):
                            cv2.circle(img, (int(particle_projs[pi, 0]), int(particle_projs[pi, 1])), 3, (0, 0, 255), -1)
                        # project eef particle
                        eef_cam = opengl2cam(eef_kp[0], extr)
                        eef_proj = np.zeros((1, 2))
                        eef_proj[0, 0] = eef_cam[0, 0] * fx / eef_cam[0, 2] + cx
                        eef_proj[0, 1] = eef_cam[0, 1] * fy / eef_cam[0, 2] + cy
                        cv2.circle(img, (int(eef_proj[0, 0]), int(eef_proj[0, 1])), 3, (0, 255, 0), -1)
                        # print(eef_kp[0].mean(0), eef_proj.mean(0))
                        cv2.imwrite(f'test_{episode_idx}_{cam_idx}_{curr_frame}.jpg', img)
                        return
                
                elif dist_curr >= dist_thresh_range[1]:
                    break

                fi += 1

        frame_idxs = np.array(frame_idxs)
        np.savetxt(os.path.join(frame_idx_dir, f'{episode_idx}.txt'), frame_idxs, fmt='%d')
        print(f'episode {episode_idx} has {cnt} pushes')


if __name__ == "__main__":
    args = gen_args()
    _ = gen_model(args, material_dict=None, debug=True)

    data_dir_list = [
        "../data/rope-new",
    ]
    for data_dir in data_dir_list:
        if os.path.isdir(data_dir):
            extract_pushes(args, data_dir, save_dir=data_dir)
