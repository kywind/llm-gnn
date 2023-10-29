import abc
import glob
import json
import os
import random
import re
import pickle
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import pickle as pkl

from data.utils import depth2fgpcd, np2o3d, fps_rad_idx
from preprocess_rope_2 import extract_pushes
from dgl.geometry import farthest_point_sampler



def construct_edges_from_states(states, adj_thresh, mask, eef_mask, no_self_edge=False):  # helper function for construct_graph
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N) torch tensor, true when index is a valid particle
    # :param eef_mask: (B, N) torch tensor, true when index is a valid eef particle
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor
    B, N, state_dim = states.shape
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    threshold = adj_thresh * adj_thresh
    dis = torch.sum((s_sender - s_receiv)**2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask = mask_1 * mask_2
    dis[~mask] = 1e10  # avoid invalid particles to particles relations
    eef_mask_1 = eef_mask[:, :, None].repeat(1, 1, N)
    eef_mask_2 = eef_mask[:, None, :].repeat(1, N, 1)
    eef_mask = eef_mask_1 * eef_mask_2
    dis[eef_mask] = 1e10  # avoid eef to eef relations
    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # remove self edge
    if no_self_edge:
        self_edge_mask = torch.eye(N, device=states.device, dtype=states.dtype)[None, :, :]
        adj_matrix = adj_matrix * (1 - self_edge_mask)

    # add topk constraints
    topk = 5
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    
    n_rels = adj_matrix.sum(dim=(1,2))
    n_rel = n_rels.max().long().item()
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return Rr, Rs


'''
Do particle sampling based on raw obj keypoints and eef keypoints data
Shoe pushing dataset 
'''
class CanonicalRopeDynDataset(Dataset):
    def __init__(self, 
        args,
        data_dir,
        prep_save_dir,
        ratios,
        phase='train',
        fixed_idx=False,
        dist_thresh_range=[0.02, 0.05],  # for selecting pushes during preprocessing
        n_future=1,
        # adj_thresh_range=[0.2, 0.3],  # for constructing edges
    ):
        self.args = args
        self.phase = phase
        self.data_dir = data_dir
        print(f'Setting up CanonicalRopeDynDataset')
        print(f'Setting up {phase} dataset, data_dir: {len(data_dir)}')

        self.fixed_idx = fixed_idx
        self.fps_idx_list = []  # to be filled in __getitem__
        self.n_future = n_future
        self.dist_thresh_range = dist_thresh_range
        # self.adj_thresh_range = adj_thresh_range

        self.max_n = 1  # max number of objects
        self.max_nobj = 100  # max number of object points
        self.max_neef = 1  # max number of eef points
        self.fps_radius = 0.1  # radius for farthest point sampling
        self.top_k = 100  # maximal top k particles to consider for first step fps

        self.obj_kypts_paths = []
        self.eef_kypts_paths = []
        self.physics_paths = []

        # preprocess
        save_dir = os.path.join(prep_save_dir, data_dir.split('/')[-1])
        if os.path.exists(save_dir):
            if not os.path.exists(os.path.join(save_dir, 'metadata.txt')):
                preprocess = True
                print(f'Found preprocess dir {save_dir} but without metadata, re-preprocessing...')
            else:
                with open(os.path.join(save_dir, 'metadata.txt'), 'r') as f:
                    metadata = f.read().strip().split(',')
                assert len(metadata) == 3, f'Wrong metadata format: {metadata}, maybe consider using older datasets'
                if float(metadata[0]) == dist_thresh_range[0] and float(metadata[1]) == dist_thresh_range[1] and int(metadata[2]) == n_future:
                    preprocess = False
                    print(f'Found preprocess dir {save_dir}, skipping preprocessing...')
                else:
                    preprocess = True
                    print(f'Found preprocess dir {save_dir} but with different metadata, re-preprocessing...')
        else:
            preprocess = True
            print(f'No preprocess dir {save_dir}, preprocessing...')
        if preprocess:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
                f.write(f'{dist_thresh_range[0]},{dist_thresh_range[1]},{n_future}')
            extract_pushes(args,
                data_dir=data_dir, 
                save_dir=save_dir, 
                dist_thresh_range=dist_thresh_range)

        # load kypts paths
        num_episodes = len(list(glob.glob(os.path.join(data_dir, f"episode_*"))))
        print(f"Found num_episodes: {num_episodes}")
        frame_count = 0
        for episode_idx in range(num_episodes):
            n_frames = len(list(glob.glob(os.path.join(data_dir, f"episode_{episode_idx}/camera_0/*_particles.npy"))))
            obj_kypts_paths = [os.path.join(data_dir, f"episode_{episode_idx}/camera_0", f"{frame_idx}_particles.npy") for frame_idx in range(n_frames)]
            eef_kypts_paths = [os.path.join(data_dir, f"episode_{episode_idx}/camera_0", f"{frame_idx}_endeffector.npy") for frame_idx in range(n_frames)]
            physics_path = os.path.join(data_dir, f"episode_{episode_idx}/property.json")
            self.obj_kypts_paths.append(obj_kypts_paths)
            self.eef_kypts_paths.append(eef_kypts_paths)
            self.physics_paths.append(physics_path)
            frame_count += n_frames
        print(f'Found {frame_count} frames in {data_dir}')

        # load data pairs
        pairs_path = os.path.join(save_dir, 'frame_pairs')
        self.pair_lists = []
        for episode_idx in range(num_episodes):
            prev_pair_len = len(self.pair_lists)
            frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}.txt'))
            if len(frame_pairs.shape) == 1: continue
            push_start = frame_pairs[:, 0]
            push_end = frame_pairs[:, 1]
            pairs = [(episode_idx, int(start), int(end)) for start, end in zip(push_start, push_end)]
            self.pair_lists.extend(pairs)
            curr_pair_len = len(self.pair_lists)
        self.pair_lists = np.array(self.pair_lists)
        print(f'Found {len(self.pair_lists)} frame pairs in {pairs_path}')

        # load physics params
        self.physics_params = []
        for episode_idx in range(num_episodes):
            physics_path = self.physics_paths[episode_idx]
            assert os.path.join(self.data_dir, f"episode_{episode_idx}/property.json") == physics_path
            with open(physics_path) as f:
                properties = json.load(f)
            physics_param = np.array([
                properties['particle_radius'] * 10,
                properties['num_particles'] * 0.001,
                properties['length'] * 0.05,
                properties['thickness'] * 0.02,
                properties['dynamic_friction'] * 2,
                properties['cluster_spacing'] * 0.1,
                properties['global_stiffness'] * 1,
            ]).astype(np.float32)  # approximate normalization
            self.physics_params.append(physics_param)
        self.physics_params = np.stack(self.physics_params, axis=0)  # (N, phys_dim)

        self.ratio = ratios[phase]
        print(f'Taking ratio {self.ratio} of {len(self.pair_lists)} eef keypoint files')
        self.pair_lists = self.pair_lists[int(len(self.pair_lists) * self.ratio[0]):int(len(self.pair_lists) * self.ratio[1])]
        print(f'{phase} dataset has {len(self.pair_lists)} pairs')

        self.pair_lists_episode_idx = {}
        prev_pair_len = 0
        episode_cnt = 0
        for episode_idx in range(num_episodes):
            episode_len = len(self.pair_lists[self.pair_lists[:, 0] == episode_idx])
            if episode_len != 0:
                episode_cnt += 1
            curr_pair_len = episode_len + prev_pair_len
            self.pair_lists_episode_idx[episode_idx] = [prev_pair_len, curr_pair_len]
            prev_pair_len = curr_pair_len
        assert prev_pair_len == len(self.pair_lists)
        print(f'{phase} dataset has {episode_cnt} episodes')
    
        self.epoch_ratio = 0.01 if phase == 'train' else 0.01
        print(f'{phase} dataset is using {int(len(self.pair_lists) * self.epoch_ratio)} pairs per epoch')
    
    def reset_epoch(self):
        if self.phase == 'train':
            self.epoch_idx_list = np.random.choice(len(self.pair_lists), size=int(len(self.pair_lists) * self.epoch_ratio), replace=False)
        else:
            self.epoch_idx_list = np.linspace(0, len(self.pair_lists) - 1, num=int(len(self.pair_lists) * self.epoch_ratio), dtype=int)

    def __len__(self):
        return int(len(self.pair_lists) * self.epoch_ratio)

    def __getitem__(self, i):
        time1 = time.time()
        i = self.epoch_idx_list[i]
        args = self.args
        max_n = self.max_n
        max_nobj = self.max_nobj
        max_neef = self.max_neef
        n_future = self.n_future
        fps_radius = self.fps_radius

        pair = self.pair_lists[i]
        episode_idx, start, end = pair

        # get keypoints
        obj_kp_path_start = self.obj_kypts_paths[episode_idx][start]
        obj_kp_path_end = self.obj_kypts_paths[episode_idx][end]
        eef_kp_path_start = self.eef_kypts_paths[episode_idx][start]
        eef_kp_path_end = self.eef_kypts_paths[episode_idx][end]
        # sanity check
        assert os.path.join(self.data_dir, f"episode_{episode_idx}/camera_0", f"{start}_particles.npy") == obj_kp_path_start

        # load obj kypts
        obj_kp_start = np.load(obj_kp_path_start)[:, :3].astype(np.float32)   # (N, 3)
        obj_kp_end = np.load(obj_kp_path_end)[:, :3].astype(np.float32)   # (N, 3)
        # multi object list
        obj_kp_start = [obj_kp_start]
        obj_kp_end = [obj_kp_end]

        # load eef kypts
        eef_kp_start = np.load(eef_kp_path_start).astype(np.float32)   # (3)
        eef_kp_end = np.load(eef_kp_path_end).astype(np.float32)   # (3)
        x_start = eef_kp_start[0]
        z_start = eef_kp_start[1]
        x_end = eef_kp_end[0]
        z_end = eef_kp_end[1]
        # sanity check
        dist_curr = np.sqrt((x_start - x_end) ** 2 + (z_start - z_end) ** 2)
        assert dist_curr >= self.dist_thresh_range[0] and dist_curr < self.dist_thresh_range[1]
        # concatenate eef kypts with y randomness
        y = np.mean(obj_kp_start[0][:, 1])
        y_noise_1 = np.random.uniform(-0.005, 0.005)
        y_noise_2 = np.random.uniform(-0.005, 0.005)
        eef_kp = np.array([[[x_start, y + y_noise_1, -z_start]], [[x_end, y + y_noise_2, -z_end]]], dtype=np.float32)  # (2, 1, 3)
        eef_kp_num = eef_kp.shape[1]

        # TODO replace this sampling to a random version
        if not self.fixed_idx or (self.fixed_idx and self.fps_idx_list == []):
            self.fps_idx_list = []
            for j in range(len(obj_kp_start)):
                # farthest point sampling
                particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...]
                fps_idx_tensor = farthest_point_sampler(particle_tensor, self.top_k, start_idx=np.random.randint(0, obj_kp_start[j].shape[0]))[0]
                fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)

                # downsample to uniform radius
                downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
                _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
                fps_idx_2 = fps_idx_2.astype(int)
                fps_idx = fps_idx_1[fps_idx_2]
                # fps_idx = fps_idx_1
                # print(fps_idx_1.shape, fps_idx_1.max(), fps_idx_1.dtype, fps_idx_2.shape, fps_idx_2.max(), fps_idx_2.dtype)
                self.fps_idx_list.append(fps_idx)
        obj_kp_start = [obj_kp_start[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
        instance_num = len(obj_kp_start)
        assert instance_num == 1
        obj_kp_start = np.concatenate(obj_kp_start, axis=0) # (N, 3)
        obj_kp_num = obj_kp_start.shape[0]

        obj_kp_end = [obj_kp_end[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
        obj_kp_end = np.concatenate(obj_kp_end, axis=0) # (N, 3)
        assert obj_kp_end.shape[0] == obj_kp_num

        if max_nobj is not None:
            # pad obj_kp_start
            obj_kp_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            obj_kp_pad[:obj_kp_num] = obj_kp_start
            obj_kp_start = obj_kp_pad
            # pad obj_kp_end
            obj_kp_end_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            obj_kp_end_pad[:obj_kp_num] = obj_kp_end
            obj_kp_end = obj_kp_end_pad
        else:
            print("not using max_nobj which only works for fixed number of obj keypoints")
            max_nobj = obj_kp_num

        # add more future states
        obj_kp_futures = [obj_kp_end]
        obj_future_mask = [True]
        if n_future > 1:
            current_start = start
            current_end = end
            pair_list = [(episode_idx, current_start, current_end)]
            for fi in range(1, n_future):
                # search for next frame
                # accelerate search by only searching in the same episode
                pair_episode_start, pair_episode_end = self.pair_lists_episode_idx[episode_idx]
                pairs = self.pair_lists[pair_episode_start:pair_episode_end]
                valid_pairs_future = pairs[pairs[:, 1] == current_end]
                if len(valid_pairs_future) == 0:
                    # exceed the dataset
                    obj_kp_futures.extend([obj_kp_end] * (n_future - fi))
                    obj_future_mask.extend([False] * (n_future - fi))
                    pair_list.extend([(episode_idx, current_end, current_end) for _ in range(n_future - fi)])
                    break
                
                random_idx = np.random.randint(0, len(valid_pairs_future))
                current_start = valid_pairs_future[random_idx][1]
                current_end = valid_pairs_future[random_idx][2]
                pair_list.append((episode_idx, current_start, current_end))

                obj_kp_path_future = self.obj_kypts_paths[episode_idx][current_end]
                obj_kp_future = np.load(obj_kp_path_future)[:, :3].astype(np.float32)   # (N, 3)
                # multi object list
                obj_kp_future = [obj_kp_future]
                # downsample
                obj_kp_future = [obj_kp_future[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]

                obj_kp_future = np.concatenate(obj_kp_future, axis=0) # (N, 3)
                assert obj_kp_future.shape[0] == obj_kp_num

                if max_nobj is not None:
                    # pad obj_kp_future
                    obj_kp_future_pad = np.zeros((max_nobj, 3), dtype=np.float32)
                    obj_kp_future_pad[:obj_kp_num] = obj_kp_future
                    obj_kp_future = obj_kp_future_pad
                assert obj_kp_future.shape[0] == max_nobj

                obj_kp_futures.append(obj_kp_future)
                obj_future_mask.append(True)

        obj_kp_futures = np.stack(obj_kp_futures, axis=0)  # (n_future, N, 3)
        obj_future_mask = np.array(obj_future_mask, dtype=bool)  # (n_future,)

        if max_neef is not None:
            # pad eef_kp
            eef_kp_pad = np.zeros((2, max_neef, 3), dtype=np.float32)
            eef_kp_pad[:, :eef_kp_num] = eef_kp
            eef_kp = eef_kp_pad
        else:
            print("not using max_neef which only works for fixed number of eef keypoints")
            max_neef = eef_kp_num
        
        if max_n is not None:
            assert max_n >= instance_num
        else:
            print("not using max_n which only works for fixed number of objects")
            max_n = instance_num

        state_mask = np.zeros((max_nobj + max_neef), dtype=bool)
        state_mask[max_nobj : max_nobj + eef_kp_num] = True
        state_mask[:obj_kp_num] = True

        eef_mask = np.zeros((max_nobj + max_neef), dtype=bool)
        eef_mask[max_nobj : max_nobj + eef_kp_num] = True

        obj_mask = np.zeros((max_nobj,), dtype=bool)
        obj_mask[:obj_kp_num] = True

        # construct instance information
        p_rigid = np.zeros(max_n, dtype=np.float32)  # clothes are nonrigid
        p_instance = np.zeros((max_nobj, max_n), dtype=np.float32)
        j_perm = np.random.permutation(instance_num)
        ptcl_cnt = 0
        # sanity check
        assert sum([len(self.fps_idx_list[j]) for j in range(len(self.fps_idx_list))]) == obj_kp_num
        # fill in p_instance
        for j in range(instance_num):
            p_instance[ptcl_cnt:ptcl_cnt + len(self.fps_idx_list[j_perm[j]]), j_perm[j]] = 1
            ptcl_cnt += self.fps_idx_list[j_perm[j]]

        # construct physics information
        physics_param = np.tile(self.physics_params[episode_idx], (max_nobj, 1))  # (N, phys_dim)

        # construct attributes
        attr_dim = 2
        assert attr_dim == args.attr_dim
        attrs = np.zeros((max_nobj + max_neef, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + eef_kp_num, 1] = 1.

        # construct state
        state_start = np.concatenate([obj_kp_start, eef_kp[0]], axis=0)  # (N+M, 3)

        # action encoded as state_delta (only stored in eef keypoints)
        states_delta = np.zeros((max_nobj + max_neef, state_start.shape[-1]), dtype=np.float32)
        states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

        # init future states and actions
        eef_future = np.zeros((n_future - 1, max_nobj + max_neef, state_start.shape[-1]), dtype=np.float32)
        states_delta_future = np.zeros((n_future - 1, max_nobj + max_neef, state_start.shape[-1]), dtype=np.float32)
        # construct future states and actions
        for fi in range(n_future - 1):
            _, future_start, future_end = pair_list[fi]
            # load eef kypts
            eef_kp_path_future_start = self.eef_kypts_paths[episode_idx][future_start]
            eef_kp_path_future_end = self.eef_kypts_paths[episode_idx][future_end]
            eef_kp_future_start = np.load(eef_kp_path_future_start).astype(np.float32)   # (3)
            eef_kp_future_end = np.load(eef_kp_path_future_end).astype(np.float32)   # (3)
            x_future_start = eef_kp_future_start[0]
            z_future_start = eef_kp_future_start[1]
            x_future_end = eef_kp_future_end[0]
            z_future_end = eef_kp_future_end[1]

            y_mean = np.mean(obj_kp_futures[fi][:, 1])  # mean of last step end frame particles y
            y_noise_1 = np.random.uniform(-0.005, 0.005)
            y_noise_2 = np.random.uniform(-0.005, 0.005)
            eef_kp_future = np.array([[[x_future_start, y_mean + y_noise_1, -z_future_start]], [[x_future_end, y_mean + y_noise_2, -z_future_end]]], dtype=np.float32)
            eef_future[fi, max_nobj : max_nobj + eef_kp_num] = eef_kp_future[0]
            states_delta_future[fi, max_nobj : max_nobj + eef_kp_num] = eef_kp_future[1] - eef_kp_future[0]

        # history state
        state_start = state_start[None]  # n_his = 1  # TODO make this code compatible with n_his > 1

        # add randomness
        # state randomness
        state_start += np.random.uniform(-0.005, 0.005, size=state_start.shape)  # TODO tune noise level
        # rotation randomness (already translation-invariant)
        random_rot = np.random.uniform(-np.pi, np.pi)
        rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                            [np.sin(random_rot), np.cos(random_rot), 0],
                            [0, 0, 1]], dtype=state_start.dtype)  # 2D rotation matrix in xy plane
        state_start = state_start @ rot_mat[None]
        states_delta = states_delta @ rot_mat
        eef_future = eef_future @ rot_mat[None]
        states_delta_future = states_delta_future @ rot_mat[None]
        obj_kp_futures = obj_kp_futures @ rot_mat[None]

        # save graph
        graph = {
            # input information
            "state": state_start,  # (n_his, N+M, state_dim)
            "action": states_delta,  # (N+M, state_dim)

            # future information
            "eef_future": eef_future,  # (n_future, N+M, state_dim)
            "action_future": states_delta_future,  # (n_future, N+M, state_dim)

            # ground truth information
            "state_future": obj_kp_futures,  # (n_future, N, state_dim)
            "state_future_mask": obj_future_mask,  # (n_future,)

            # relation information
            # "Rr": Rr,  # (n_rel, N+M)
            # "Rs": Rs,  # (n_rel, N+M)

            # attr information
            "attrs": attrs,  # (N+M, attr_dim)
            "p_rigid": p_rigid,  # (n_instance,)
            "p_instance": p_instance,  # (N, n_instance)
            "physics_param": physics_param,  # (N, phys_dim)
            "state_mask": state_mask,  # (N+M,)
            "eef_mask": eef_mask,  # (N+M,)
            "obj_mask": obj_mask,  # (N,)
        }
        # print([f"{key}: {val.shape}" for key, val in graph.items()])
        time2 = time.time()
        # print(f'{time2 - time1:.6f}')
        return graph
