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


def extract_pushes(args, data_dir, save_dir, dist_thresh=0.05):
    camera_indices = [0, 1, 2, 3]
    cam_idx = 0  # follow camera 1's particles
    fixed_idx = True  # use same sampling indices for all frames
    fps_idx = None

    # n_particle = 100
    push_thres = dist_thresh

    # dense keypoints
    eef_kypts_dir = os.path.join(save_dir, 'kypts', 'eef_kypts') # list of (2, eef_ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(eef_kypts_dir, exist_ok=True)
    obj_kypts_dir = os.path.join(save_dir, 'kypts', 'obj_kypts') # list of (ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(obj_kypts_dir, exist_ok=True)
    physics_dir = os.path.join(save_dir, 'kypts', 'physics')
    os.makedirs(physics_dir, exist_ok=True)

    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))

    num_frames_total = 0
    ep_fr_list = []
    actions_total = []
    steps_total = []
    properties_total = []
    for episode_idx in range(num_episodes):
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/*_color.png"))))
        num_frames_total += num_frames
        ep_fr_list.extend([(episode_idx, fi) for fi in range(num_frames)])

        actions = np.load(os.path.join(data_dir, f"episode_{episode_idx}/actions.npy"))
        steps = np.load(os.path.join(data_dir, f"episode_{episode_idx}/steps.npy"))
        actions_total.append(actions)
        steps_total.append(steps)

        with open(os.path.join(data_dir, f"episode_{episode_idx}/property.json")) as f:
            properties = json.load(f)
        properties_total.append(properties)

    curr_step = None
    curr_frame = -1
    curr_episode = -1
    for (episode_idx, fi) in ep_fr_list:
        # if episode_idx > 0: raise Exception
        # if episode_idx == 19 or episode_idx == 26: continue
        if curr_episode != episode_idx:
            curr_episode = episode_idx
            curr_step = None
        actions = actions_total[episode_idx]
        steps = steps_total[episode_idx]

        if len(actions) != len(steps):
            continue

        steps_a = np.concatenate([[0], steps], axis=0)
        new_step = None
        for si in range(len(steps_a) - 1):
            # if step is [x1, x2]: (0, x1-1) == [1, x1-2] is first push, (x1+2, x2-1) == [x1+1, x2-2] is second push
            if fi >= steps_a[si] + 1 and fi <= steps_a[si+1] - 2:
                new_step = si
                break
        else:
            continue  # this frame is not valid
        assert new_step is not None

        if curr_step is None or new_step != curr_step:
            # this is the start of a dense push
            curr_step = new_step
            curr_frame = fi
            continue

        start_frame = steps_a[curr_step] + 1
        end_frame = steps_a[curr_step + 1] - 2  # inclusive

        action = actions[curr_step]
        x_start, z_start, x_end, z_end = action
        dist = np.sqrt((x_start - x_end) ** 2 + (z_start - z_end) ** 2)

        # preprocessing start and end frame (only for v1 data)
        preprocess_frames = True
        if preprocess_frames:
            from scipy.spatial.distance import cdist
            speed = 1.0 / 100
            n_frames_push = int(dist / speed) + 1

            # particles before each push
            particles_before = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{start_frame - 1}_particles.npy"))
            for fii in range(n_frames_push):
                xx = x_start + (x_end - x_start) * fii / (n_frames_push - 1)
                zz = z_start + (z_end - z_start) * fii / (n_frames_push - 1)
                robot_obj_dist = np.min(cdist(np.array([[xx, -zz]]), particles_before[:, [0, 2]]))
                if robot_obj_dist < 0.2:
                    # print(fii, n_frames_push)
                    x_start = xx
                    z_start = zz
                    x_end = xx + (x_end - x_start) * (fii + end_frame - start_frame) / (n_frames_push - 1)
                    z_end = zz + (z_end - z_start) * (fii + end_frame - start_frame) / (n_frames_push - 1)
                    dist = np.sqrt((x_start - x_end) ** 2 + (z_start - z_end) ** 2)
                    break
            # else:
            #     import ipdb; ipdb.set_trace()

            # speed = dist / (end_frame - start_frame)

        # example properties:
        # 0
        # {"particle_radius": 0.025, 
        # "num_particles": 709, 
        # "length": 69.91487868831923, 
        # "thickness": 111.07030729147404, 
        # "dynamic_friction": 0.407033170114787, 
        # "cluster_spacing": 5.190731322960941, 
        # "global_stiffness": 0.0}

        # if fi - curr_frame >= 7:  # sanity test, assume each push is uniformly 7 frames long
        #     print(curr_episode, curr_step, curr_frame, fi)
        #     curr_frame = fi
        # continue

        x_curr = x_start + (x_end - x_start) * (curr_frame - start_frame) / (end_frame - start_frame)
        z_curr = z_start + (z_end - z_start) * (curr_frame - start_frame) / (end_frame - start_frame)
        x_fi = x_start + (x_end - x_start) * (fi - start_frame) / (end_frame - start_frame)
        z_fi = z_start + (z_end - z_start) * (fi - start_frame) / (end_frame - start_frame)

        # if fi - curr_frame >= n_frames:
        dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2)
        particle_length = None
        if dist_curr >= push_thres or (fi == end_frame and curr_frame < fi):
            # print(curr_episode, curr_step, curr_frame, fi, round(dist, 2), round(dist_curr, 2))

            # get particle
            particle = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{curr_frame}_particles.npy"))
            if particle_length is None: particle_length = particle.shape[0]
            else: assert particle_length == particle.shape[0], "particle length not consistent"
            particle_next = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{fi}_particles.npy"))

            # farthest sampling to get a set of particles for current frame
            # if (fixed_idx and fps_idx is None) or not fixed_idx:
            #     particle_tensor = torch.from_numpy(particle).float()[None, ...]
            #     assert particle_tensor.shape[1] >= n_particle
            #     fps_idx_tensor = farthest_point_sampler(particle_tensor, n_particle)[0]
            #     fps_idx = fps_idx_tensor.numpy().astype(np.int32)

            fps_idx = np.arange(particle.shape[0])

            particle = particle[fps_idx, :3]
            particle_next = particle_next[fps_idx , :3]

            # save obj keypoints
            # assert particle.shape[0] == 100
            # assert particle_next.shape[0] == 100
            obj_kp = np.stack([particle, particle_next], axis=0)  # (2, n_particle, 3)
            save_path = os.path.join(obj_kypts_dir, f'{episode_idx:03}_{curr_frame:03}.npy')
            np.save(save_path, obj_kp)

            # get action for current frame
            y = 0.5
            pos_start = np.array([[x_curr, y, z_curr]])  # (1, 3)
            pos_end = np.array([[x_fi, y, z_fi]])
            eef_kp = np.stack([pos_start, pos_end], axis=0)  # (2, 1, 3)
            eef_kp[:, :, 2] *= -1

            # save eef keypoints
            save_path = os.path.join(eef_kypts_dir, f'{episode_idx:03}_{curr_frame:03}.npy')
            np.save(save_path, eef_kp)

            # save physics
            prop = properties_total[episode_idx]
            prop_np = np.array([
                prop['particle_radius'] * 10,
                prop['num_particles'] * 0.001,
                prop['length'] * 0.05,
                prop['thickness'] * 0.02,
                prop['dynamic_friction'] * 2,
                prop['cluster_spacing'] * 0.1,
                prop['global_stiffness'] * 1,
            ])  # approximate normalization
            save_path = os.path.join(physics_dir, f'{episode_idx:03}_{curr_frame:03}.npy')
            np.save(save_path, prop_np)

            img_vis = False
            if img_vis:
                intr = np.load(os.path.join(data_dir, f"camera_intrinsic_params.npy"))[cam_idx]
                extr = np.load(os.path.join(data_dir, f"camera_extrinsic_matrix.npy"))[cam_idx]
                img = cv2.imread(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{curr_frame}_color.png"))
                # transform particle to camera coordinate
                particle_cam = opengl2cam(particle, extr)
                # particle_cam = (extr @ np.concatenate([particle, np.ones((particle.shape[0], 1))], axis=1).T).T
                # particle_cam[:, 1] *= -1
                # particle_cam[:, 2] *= -1
                assert particle_cam[:, 2].min() > 0  # z > 0
                # project particle
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
                # return

            curr_frame = fi


if __name__ == "__main__":
    args = gen_args()
    _ = gen_model(args, material_dict=None, debug=True)

    data_dir_list = [
        "../data/rope",
    ]
    for data_dir in data_dir_list:
        if os.path.isdir(data_dir):
            extract_pushes(args, data_dir)
