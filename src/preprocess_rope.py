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


def load_particles(args, data_dir):
    camera_indices = [0, 1, 2, 3]
    episode_idx = 0
    num_frames = len(list(glob.glob(os.path.join(data_dir, f"camera_1/episode_{episode_idx}/*_color.png"))))

    # get point merge mask
    # for fi in range(num_frames - 1):
    for fi in range(1):
        particle_test_list = []
        for cam_idx in camera_indices:
            particle = np.load(os.path.join(data_dir, f"camera_{cam_idx}/episode_{episode_idx}/{fi}_particles.npy"))[:, :3]
            particle_test_list.append(particle)
        particle_test_list = np.stack(particle_test_list, axis=0) # (num_frames, num_particles, 3)

        # rand_index = np.random.choice(particle_test_list.shape[1], 100)
        # particle_test_list = particle_test_list[:, rand_index, :3]

        cam_img_vis = False
        if cam_img_vis:
            for ci, cam_idx in enumerate(camera_indices):
                intr = np.load(os.path.join(data_dir, f"camera_{cam_idx}/camera_intrinsic_params.npy"))
                extr = np.load(os.path.join(data_dir, f"camera_{cam_idx}/camera_extrinsic_matrix.npy"))
                img = cv2.imread(os.path.join(data_dir, f"camera_{cam_idx}/episode_{episode_idx}/{fi}_color.png"))

                particle = particle_test_list[ci]
                # transform particle to camera coordinate
                particle_cam = opengl2cam(particle, extr)
                # project particle
                fx, fy, cx, cy = intr
                particle_projs = np.zeros((particle_cam.shape[0], 2))
                particle_projs[:, 0] = particle_cam[:, 0] * fx / particle_cam[:, 2] + cx
                particle_projs[:, 1] = particle_cam[:, 1] * fy / particle_cam[:, 2] + cy

                # transform particle to camera coordinate
                particle_cam = opengl2cam(particle, extr)
                # project particle
                fx, fy, cx, cy = intr
                particle_projs = np.zeros((particle_cam.shape[0], 2))
                particle_projs[:, 0] = particle_cam[:, 0] * fx / particle_cam[:, 2] + cx
                particle_projs[:, 1] = particle_cam[:, 1] * fy / particle_cam[:, 2] + cy

                for pi in range(particle_projs.shape[0]):
                    cv2.circle(img, (int(particle_projs[pi, 0]), int(particle_projs[pi, 1])), 3, (0, 0, 255), -1)
                cv2.imwrite(f'test_{cam_idx}.jpg', img)

        p1 = None
        p2 = None
        dist_list = np.zeros((len(camera_indices) - 1, particle_test_list.shape[1]))
        for ci, cam_idx in enumerate(camera_indices):
            if ci == 0:
                p1 = particle_test_list[ci]
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(p1)
            else:
                p2 = particle_test_list[ci]
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(p2)

                # chamfer distance
                dist1 = pcd1.compute_point_cloud_distance(pcd2)
                dist1 = np.asarray(dist1)
                # dist2 = pcd2.compute_point_cloud_distance(pcd1)
                # dist2 = np.asarray(dist2)
                # dist = (dist1 + dist2) / 2
                # dist_list[ci - 1] = dist
                dist_list[ci - 1] = dist1
        
        dist_max = dist_list.max(0)
        dist_max_median = np.median(dist_max)
        valid_particle_mask = dist_max < dist_max_median
        # valid_particle_mask = dist_max < 0

        cam_valid_img_vis = True
        if cam_valid_img_vis:
            particle_test_list = particle_test_list[:, valid_particle_mask]
            for ci, cam_idx in enumerate(camera_indices):
                intr = np.load(os.path.join(data_dir, f"camera_{cam_idx}/camera_intrinsic_params.npy"))
                extr = np.load(os.path.join(data_dir, f"camera_{cam_idx}/camera_extrinsic_matrix.npy"))
                img = cv2.imread(os.path.join(data_dir, f"camera_{cam_idx}/episode_{episode_idx}/{fi}_color.png"))

                particle = particle_test_list[ci]
                # transform particle to camera coordinate
                particle_cam = opengl2cam(particle, extr)
                # project particle
                fx, fy, cx, cy = intr
                particle_projs = np.zeros((particle_cam.shape[0], 2))
                particle_projs[:, 0] = particle_cam[:, 0] * fx / particle_cam[:, 2] + cx
                particle_projs[:, 1] = particle_cam[:, 1] * fy / particle_cam[:, 2] + cy

                # transform particle to camera coordinate
                particle_cam = opengl2cam(particle, extr)
                # project particle
                fx, fy, cx, cy = intr
                particle_projs = np.zeros((particle_cam.shape[0], 2))
                particle_projs[:, 0] = particle_cam[:, 0] * fx / particle_cam[:, 2] + cx
                particle_projs[:, 1] = particle_cam[:, 1] * fy / particle_cam[:, 2] + cy

                for pi in range(particle_projs.shape[0]):
                    cv2.circle(img, (int(particle_projs[pi, 0]), int(particle_projs[pi, 1])), 3, (0, 0, 255), -1)
                cv2.imwrite(f'test_{cam_idx}.jpg', img)


def extract_pushes(args, data_dir, max_n=None, max_nobj=None, max_neef=None, max_nR=None):
    camera_indices = [0, 1, 2, 3]
    cam_idx = 0  # follow camera 1's particles
    fixed_idx = False  # use same sampling indices for all frames
    fps_idx = None


    eef_kypts_dir = os.path.join(data_dir, 'kypts', 'eef_kypts') # list of (2, eef_ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(eef_kypts_dir, exist_ok=True)
    obj_kypts_dir = os.path.join(data_dir, 'kypts', 'obj_kypts') # list of (ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(obj_kypts_dir, exist_ok=True)

    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))

    for episode_idx in range(num_episodes):
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/*_color.png"))))
        
        actions = np.load(os.path.join(data_dir, f"episode_{episode_idx}/actions.npy"))
        steps = np.load(os.path.join(data_dir, f"episode_{episode_idx}/steps.npy"))
        with open(os.path.join(data_dir, f"episode_{episode_idx}/property.json")) as f:
            properties = json.load(f)
        
        # example properties:
        # 0
        # {"particle_radius": 0.025, 
        # "num_particles": 709, 
        # "length": 69.91487868831923, 
        # "thickness": 111.07030729147404, 
        # "dynamic_friction": 0.407033170114787, 
        # "cluster_spacing": 5.190731322960941, 
        # "global_stiffness": 0.0}
        # 1
        # {"particle_radius": 0.025, 
        # "num_particles": 709, 
        # "length": 76.35750919393547, 
        # "thickness": 131.923677920146, 
        # "dynamic_friction": 0.6962350700829444, 
        # "cluster_spacing": 5.461878409601606, 
        # "global_stiffness": 0.0}
        # 76
        # {"particle_radius": 0.025, 
        # "num_particles": 1208, 
        # "length": 96.91241246432143, 
        # "thickness": 119.65478504769021, 
        # "dynamic_friction": 0.39641859421409664, 
        # "cluster_spacing": 4.810066265407364, 
        # "global_stiffness": 0.0}

        # each frame is an entire push
        delta_fi = 1
        for fi in range(num_frames - 1):
            # get particle for current frame
            particle = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{fi}_particles.npy"))

            # get particle for next frame
            particle_next = np.load(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{fi+delta_fi}_particles.npy"))

            # farthest sampling to get a set of particles for current frame
            if (fixed_idx and fps_idx is None) or not fixed_idx:
                n_particle = 50
                particle_tensor = torch.from_numpy(particle).float()[None, ...]
                fps_idx_tensor = farthest_point_sampler(particle_tensor, n_particle)[0]
                fps_idx = fps_idx_tensor.numpy().astype(np.int32)

            particle = particle[fps_idx, :3]
            particle_next = particle_next[fps_idx, :3]

            
            img_vis = True
            if img_vis:
                intr = np.load(os.path.join(data_dir, f"camera_intrinsic_params.npy"))[cam_idx]
                extr = np.load(os.path.join(data_dir, f"camera_extrinsic_matrix.npy"))[cam_idx]
                img = cv2.imread(os.path.join(data_dir, f"episode_{episode_idx}/camera_{cam_idx}/{fi}_color.png"))
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
                cv2.imwrite(f'test_{fi}.jpg', img)
                return

            # save obj keypoints
            obj_kp = np.stack([particle, particle_next], axis=0)  # (2, n_particle, 3)
            save_path = os.path.join(obj_kypts_dir, f'{episode_idx:03}_{fi:03}.npy')
            np.save(save_path, obj_kp)

            # get action for current frame
            action_frame = actions[fi]  # 4-dim
            start = action_frame[:2]
            end = action_frame[2:]
            y = 0.5
            pos_start = np.array([[start[0], y, start[1]]])  # (1, 3)
            pos_end = np.array([[end[0], y, end[1]]])
            eef_kp = np.stack([pos_start, pos_end], axis=0)  # (2, 3)

            # save eef keypoints
            save_path = os.path.join(eef_kypts_dir, f'{episode_idx:03}_{fi:03}.npy')
            np.save(save_path, eef_kp)


if __name__ == "__main__":
    args = gen_args()
    _ = gen_model(args, material_dict=None, debug=True)

    data_dir_list = [
        "../data/rope",
    ]
    for data_dir in data_dir_list:
        if os.path.isdir(data_dir):
            extract_pushes(args, data_dir)
