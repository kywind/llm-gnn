import os
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl
import open3d as o3d
from dgl.geometry import farthest_point_sampler

from config import gen_args
from data.utils import label_colormap, opengl2cam
from preprocess import construct_edges_from_states
from gnn.model_wrapper import gen_model


def load_particles(args, data_dir):
    camera_indices = [1, 2, 3, 4]
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
    camera_indices = [1, 2, 3, 4]
    cam_idx = 1  # follow camera 1's particles

    eef_kypts_dir = os.path.join(data_dir, 'eef_kypts') # list of (2, eef_ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(eef_kypts_dir, exist_ok=True)
    obj_kypts_dir = os.path.join(data_dir, 'obj_kypts') # list of (ptcl_num, 3) for each push, indexed by push_num (dense)
    os.makedirs(obj_kypts_dir, exist_ok=True)
    # dense_eef_kypts_dir = os.path.join(data_dir, 'dense_eef_kypts')
    # os.makedirs(dense_eef_kypts_dir, exist_ok=True)
    # dense_obj_kypts_dir = os.path.join(data_dir, 'dense_obj_kypts')
    # os.makedirs(dense_obj_kypts_dir, exist_ok=True)

    num_episodes = len(list(glob.glob(os.path.join(data_dir, f"camera_1/episode_*"))))
    actions = np.load(os.path.join(data_dir, "actions.npy"))

    for episode_idx in range(num_episodes):
        num_frames = len(list(glob.glob(os.path.join(data_dir, f"camera_1/episode_{episode_idx}/*_color.png"))))

        # each frame is an entire push
        delta_fi = 1
        for fi in range(num_frames - 1):
            # get particle for current frame
            particle = np.load(os.path.join(data_dir, f"camera_{cam_idx}/episode_{episode_idx}/{fi}_particles.npy"))

            # get particle for next frame
            particle_next = np.load(os.path.join(data_dir, f"camera_{cam_idx}/episode_{episode_idx}/{fi+delta_fi}_particles.npy"))
            
            # farthest sampling to get a set of particles for current frame
            n_particle = 50
            particle_tensor = torch.from_numpy(particle).float()[None, ...]
            fps_idx_tensor = farthest_point_sampler(particle_tensor, n_particle)[0]
            fps_idx = fps_idx_tensor.numpy().astype(np.int32)
            particle = particle[fps_idx, :3]
            particle_next = particle_next[fps_idx, :3]

            img_vis = False
            if img_vis:
                intr = np.load(os.path.join(data_dir, f"camera_{cam_idx}/camera_intrinsic_params.npy"))
                extr = np.load(os.path.join(data_dir, f"camera_{cam_idx}/camera_extrinsic_matrix.npy"))
                img = cv2.imread(os.path.join(data_dir, f"camera_{cam_idx}/episode_{episode_idx}/{fi}_color.png"))
                # transform particle to camera coordinate
                particle_cam = opengl2cam(particle, extr)
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
            action_frame = actions[episode_idx, fi]  # 4-dim
            start = action_frame[:2]
            end = action_frame[2:]
            y = 0.5
            pos_start = np.array([[start[0], y, start[1]]])  # (1, 3)
            pos_end = np.array([[end[0], y, end[1]]])
            eef_kp = np.stack([pos_start, pos_end], axis=0)  # (2, 3)

            # save eef keypoints
            save_path = os.path.join(eef_kypts_dir, f'{episode_idx:03}_{fi:03}.npy')
            np.save(save_path, eef_kp)


def preprocess_graph(args, data_dir, max_n=None, max_nobj=None, max_neef=None, max_nR=None):  # save states, relations and attributes; use results of extract_pushes
    # data_dir = "../data/2023-09-04-18-42-27-707743/"

    save_dir = os.path.join(data_dir, "graph")
    os.makedirs(save_dir, exist_ok=True)

    obj_kp_dir = os.path.join(data_dir, "obj_kypts")
    eef_kp_dir = os.path.join(data_dir, "eef_kypts")

    obj_kp_paths = sorted(list(glob.glob(os.path.join(obj_kp_dir, '*.npy'))))
    num_datapoints = len(obj_kp_paths) - 1  # last frame do not have corresponding eef keypoints
    print("num_datapoints: ", num_datapoints)

    # all in world space, don't need to load camera params
    graph_list = []
    for i in range(num_datapoints):
        # get keypoints 
        obj_kp_path = obj_kp_paths[i]
        obj_kp = np.load(obj_kp_path) # (2, n_particle, 3)
        obj_kp_gt = obj_kp[1]  # (n_particle, 3)
        obj_kp = obj_kp[0]  # (n_particle, 3)
        obj_kp_num = obj_kp.shape[0]

        eef_kp_path = obj_kp_path.replace("obj_kypts", "eef_kypts")
        eef_kp = np.load(eef_kp_path).astype(np.float32)   # (2, 8, 3)
        eef_kp_num = eef_kp.shape[1]

        # TODO enable multiple objects
        instance_num = 1
        ptcl_per_instance = obj_kp_num
        ptcl_per_eef = eef_kp_num

        if max_nobj is not None:
            # pad obj_kp
            obj_kp_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            obj_kp_pad[:obj_kp_num] = obj_kp
            obj_kp = obj_kp_pad
        else:
            max_nobj = obj_kp_num
            # print("not using max_nobj which only works for fixed number of obj keypoints")
        
        if max_neef is not None:
            # pad eef_kp
            eef_kp_pad = np.zeros((2, max_neef, 3), dtype=np.float32)
            eef_kp_pad[:, :eef_kp_num] = eef_kp
            eef_kp = eef_kp_pad
        else:
            max_neef = eef_kp_num
            # print("not using max_neef which only works for fixed number of eef keypoints")
        
        if max_n is not None:
            max_instance_num = max_n
            assert max_instance_num >= instance_num
        else:
            # print("not using max_n which only works for fixed number of objects")
            max_instance_num = instance_num

        state_mask = np.zeros((max_nobj + max_neef), dtype=bool)
        state_mask[max_nobj : max_nobj + eef_kp_num] = True
        state_mask[:obj_kp_num] = True

        eef_mask = np.zeros((max_nobj + max_neef), dtype=bool)
        eef_mask[max_nobj : max_nobj + eef_kp_num] = True

        obj_mask = np.zeros((max_nobj,), dtype=bool)
        obj_mask[:obj_kp_num] = True

        # construct instance information
        p_rigid = np.zeros(max_instance_num, dtype=np.float32)  # clothes are nonrigid
        p_instance = np.zeros((obj_kp_num, max_instance_num), dtype=np.float32)
        for j in range(instance_num):
            p_instance[j * ptcl_per_instance : (j + 1) * ptcl_per_instance, j] = 1  # TODO different number of particles per instance
        physics_param = np.zeros(obj_kp_num, dtype=np.float32)  # 1-dim

        # construct attributes
        attr_dim = 2
        assert attr_dim == args.attr_dim
        attrs = np.zeros((max_nobj + max_neef, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + eef_kp_num, 1] = 1.

        # construct relations (density as hyperparameter)
        adj_thresh = 1
        states = np.concatenate([obj_kp, eef_kp[0]], axis=0)  # (N, 3)  # the current eef_kp
        Rr, Rs = construct_edges_from_states(torch.tensor(states).unsqueeze(0), adj_thresh, 
                                             mask=torch.tensor(state_mask).unsqueeze(0), 
                                             eef_mask=torch.tensor(eef_mask).unsqueeze(0),
                                             no_self_edge=True)
        Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()

        # action encoded as state_delta (only stored in eef keypoints)
        states_delta = np.zeros((max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
        states_delta[max_nobj : max_neef + eef_kp_num] = eef_kp[1] - eef_kp[0]

        # next state
        states_next = obj_kp_gt
        if max_nobj is not None:
            # pad states_next
            states_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            states_pad[:states_next.shape[0]] = states_next
            states_next = states_pad

        # history state
        states = states[None]  # n_his = 1  # TODO make this code compatible with n_his > 1

        # save graph
        graph = {
            "attrs": attrs, 
            "state": states, 
            "action": states_delta,
            "Rr": Rr, 
            "Rs": Rs,
            "p_rigid": p_rigid,
            "p_instance": p_instance,
            "physics_param": physics_param,
            "particle_den": np.array([adj_thresh]),
            "state_next": states_next,
            "obj_mask": obj_mask,
        }
        # print([f"{key}: {val.shape}" for key, val in graph.items()])
        graph_list.append(graph)

    if max_nR is not None:
        max_n_Rr = max_nR
        max_n_Rs = max_nR
    else:
        # print([graph['Rr'].shape[0] for graph in graph_list])
        max_n_Rr = max([graph['Rr'].shape[0] for graph in graph_list])
        max_n_Rs = max([graph['Rs'].shape[0] for graph in graph_list])

    print("max number of relations Rr: ", max_n_Rr)
    print("max number of relations Rs: ", max_n_Rs)

    for i, graph in enumerate(graph_list):
        graph["Rr"] = np.pad(graph["Rr"], ((0, max_n_Rr - graph["Rr"].shape[0]), (0, 0)), mode='constant')
        graph["Rs"] = np.pad(graph["Rs"], ((0, max_n_Rs - graph["Rs"].shape[0]), (0, 0)), mode='constant')
        save_path = os.path.join(save_dir, f'{i:06}.pkl')
        pkl.dump(graph, open(save_path, 'wb'))


if __name__ == "__main__":
    args = gen_args()
    _ = gen_model(args, material_dict=None, debug=True)

    data_dir_list = [
        "../data/shirt",
    ]
    for data_dir in data_dir_list:
        if os.path.isdir(data_dir):
            # load_particles(args, data_dir)
            extract_pushes(args, data_dir)
            preprocess_graph(args, data_dir)
