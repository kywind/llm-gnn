import abc
import glob
import json
import os
import random
import re
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import pickle as pkl

from data.utils import depth2fgpcd, np2o3d
from preprocess import extract_pushes_train
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
class MultistepRigidDynDataset(Dataset):
    def __init__(self, 
        args,
        data_dirs,
        prep_save_dir,
        ratios,
        phase='train',
        dense=True,
        fixed_idx=False,
        dist_thresh=0.02,
        n_future=1,
    ):
        self.args = args
        self.phase = phase
        self.dense = dense
        if not dense: assert n_future == 1
        if isinstance(data_dirs, dict):
            self.data_dirs = data_dirs[phase]
        elif isinstance(data_dirs, str):  # single data directory
            self.data_dirs = [data_dirs]
        print(f'Setting up {phase} dataset')
        print(f'Found {len(self.data_dirs)} data directories')

        self.dense_t_lists = []
        self.t_lists = []
        self.dir_idx_lists = []
        self.push_idx_lists = []
        self.obj_kypts_paths = []
        self.eef_kypts_paths = []
        for i, data_dir in enumerate(self.data_dirs):
            save_dir = os.path.join(prep_save_dir, data_dir.split('/')[-1])
            if os.path.exists(save_dir):
                if not os.path.exists(os.path.join(save_dir, 'metadata.txt')):
                    preprocess = True
                    print(f'Found preprocess dir {save_dir} but without metadata, re-preprocessing...')
                else:
                    with open(os.path.join(save_dir, 'metadata.txt'), 'r') as f:
                        metadata = f.read().strip().split(',')
                    if float(metadata[0]) == dist_thresh and int(metadata[1]) == n_future:
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
                    f.write(f'{dist_thresh},{n_future}')
                extract_pushes_train(args,
                    data_dir=data_dir, 
                    save_dir=save_dir, 
                    dist_thresh=dist_thresh)

            t_list_path = os.path.join(save_dir, 'push_t_list.txt')
            t_list = np.loadtxt(t_list_path)
            self.t_lists.append(t_list)
            dense_t_list_path = os.path.join(save_dir, 'push_t_list_dense.txt')
            dense_t_list = np.loadtxt(dense_t_list_path)
            self.dense_t_lists.append(dense_t_list)

            eef_kypts_dir = os.path.join(save_dir, 'dense_eef_kypts' if dense else 'eef_kypts')
            eef_kypts_path_list = sorted(glob.glob(os.path.join(eef_kypts_dir, '*.npy')))
            obj_kypts_dir = os.path.join(save_dir, 'dense_obj_kypts' if dense else 'obj_kypts')
            obj_kypts_path_list = sorted(glob.glob(os.path.join(obj_kypts_dir, '*.pkl')))
            self.obj_kypts_paths.extend(obj_kypts_path_list)
            self.eef_kypts_paths.extend(eef_kypts_path_list)
            self.dir_idx_lists.extend([i] * len(eef_kypts_path_list))
            self.push_idx_lists.extend(list(range(len(eef_kypts_path_list))))
            print(f'Found {len(obj_kypts_path_list)} obj keypoint files in {obj_kypts_dir}')
            print(f'Found {len(eef_kypts_path_list)} eef keypoint files in {eef_kypts_dir}')
            assert len(obj_kypts_path_list) == len(eef_kypts_path_list)
            if dense:
                assert len(eef_kypts_path_list) == len(dense_t_list)
            else:
                assert len(eef_kypts_path_list) == len(t_list)
        print(f'Found {len(self.obj_kypts_paths)} obj keypoint files in total')
        print(f'Found {len(self.eef_kypts_paths)} eef keypoint files in total')
        assert len(self.obj_kypts_paths) == len(self.eef_kypts_paths)
        assert len(self.dir_idx_lists) == len(self.obj_kypts_paths)
        assert len(self.push_idx_lists) == len(self.obj_kypts_paths)

        self.ratio = ratios[phase]
        print(f'Taking ratio {self.ratio} of {len(self.eef_kypts_paths)} eef keypoint files')
        self.eef_kypts_paths = self.eef_kypts_paths[int(len(self.eef_kypts_paths) * self.ratio[0]):int(len(self.eef_kypts_paths) * self.ratio[1])]
        self.obj_kypts_paths = self.obj_kypts_paths[int(len(self.obj_kypts_paths) * self.ratio[0]):int(len(self.obj_kypts_paths) * self.ratio[1])]
        print(f'{phase} dataset has {len(self.eef_kypts_paths)} eef keypoint files and {len(self.obj_kypts_paths)} obj keypoint files')

        self.max_n = 2  # max number of objects
        self.max_nobj = 40  # max number of object points
        self.max_neef = 8  # max number of eef points
        # self.max_nR = 500  # max number of relations
        self.n_future = n_future

        self.fixed_idx = fixed_idx
        self.fps_idx_list = []
        self.top_k = 20

        self.adj_thresh_range = [0.1, 0.2]


    def __len__(self):
        return len(self.eef_kypts_paths)

    
    def __getitem__(self, i):
        max_n = self.max_n
        max_nobj = self.max_nobj
        max_neef = self.max_neef
        # max_nR = self.max_nR
        n_future = self.n_future
        dense = self.dense
        args = self.args

        # get keypoints 
        obj_kp_path = self.obj_kypts_paths[i]
        eef_kp_path = self.eef_kypts_paths[i]
        # print(obj_kp_path, eef_kp_path)

        with open(obj_kp_path, 'rb') as f:
            obj_kp, obj_kp_next = pkl.load(f) # lists of (rand_ptcl_num, 3)
        eef_kp = np.load(eef_kp_path).astype(np.float32)   # (2, 8, 3)
        eef_kp_num = eef_kp.shape[1]

        # TODO replace this sampling to a random version
        if not self.fixed_idx or (self.fixed_idx and self.fps_idx_list == []):
            self.fps_idx_list = []
            for j in range(len(obj_kp)):
                # farthest point sampling
                particle_tensor = torch.from_numpy(obj_kp[j]).float()[None, ...]
                fps_idx_tensor = farthest_point_sampler(particle_tensor, self.top_k, start_idx=np.random.randint(0, obj_kp[j].shape[0]))[0]
                fps_idx = fps_idx_tensor.numpy().astype(np.int32)
                self.fps_idx_list.append(fps_idx)
        obj_kp = [obj_kp[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
        instance_num = len(obj_kp)
        ptcl_per_instance = obj_kp[0].shape[0]
        obj_kp = np.concatenate(obj_kp, axis=0) # (N = instance_num * rand_ptcl_num, 3)
        obj_kp_num = obj_kp.shape[0]

        # obj_kp_next = [kp[:top_k] for kp in obj_kp_next]
        obj_kp_next = [obj_kp_next[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
        obj_kp_next = np.concatenate(obj_kp_next, axis=0) # (N = instance_num * rand_ptcl_num, 3)
        assert obj_kp_next.shape[0] == obj_kp_num


        if max_nobj is not None:
            # pad obj_kp
            obj_kp_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            obj_kp_pad[:obj_kp_num] = obj_kp
            obj_kp = obj_kp_pad
            # pad obj_kp_next
            obj_kp_next_pad = np.zeros((max_nobj, 3), dtype=np.float32)
            obj_kp_next_pad[:obj_kp_num] = obj_kp_next
            obj_kp_next = obj_kp_next_pad
        else:
            print("not using max_nobj which only works for fixed number of obj keypoints")
            max_nobj = obj_kp_num

        # add more future states
        obj_kp_futures = [obj_kp_next]
        obj_future_mask = [True]
        # print(f"n_future: {n_future}, i: {i}")
        if n_future > 1:
            assert dense
            dir_idx_i = self.dir_idx_lists[i]
            push_idx_i = self.push_idx_lists[i]
            t_list = self.t_lists[dir_idx_i]
            dense_t_list = self.dense_t_lists[dir_idx_i]

            t_i = dense_t_list[push_idx_i][0]
            n_push_i = t_list[:, 0][t_list[:, 0] <= t_i].shape[0]
            for fi in range(n_future - 1):
                # print(f"n_future: {n_future}, fi: {fi}")
                if i + fi + 1 >= len(self.obj_kypts_paths):
                    # exceed the dataset
                    obj_kp_futures.extend([obj_kp_next] * (n_future - 1 - fi))
                    obj_future_mask.extend([False] * (n_future - 1 - fi))
                    break
                
                # get keypoints 
                obj_kp_path_future = self.obj_kypts_paths[i + fi + 1]
                with open(obj_kp_path_future, 'rb') as f:
                    obj_kp_future, _ = pkl.load(f) # lists of (rand_ptcl_num, 3)
                
                dir_idx = self.dir_idx_lists[i + fi + 1]
                push_idx = self.push_idx_lists[i + fi + 1]
                t_list = self.t_lists[dir_idx]
                dense_t_list = self.dense_t_lists[dir_idx]

                t = dense_t_list[push_idx][0]  # start of this dense push
                n_push = t_list[:, 0][t_list[:, 0] <= t].shape[0]
                # print(dir_idx, dir_idx_i, push_idx, push_idx_i, n_push, n_push_i, len(self.fps_idx_list), len(obj_kp_future), obj_kp_future[0].shape)
                if dir_idx != dir_idx_i or n_push != n_push_i:
                    # not the same push
                    obj_kp_futures.extend([obj_kp_next] * (n_future - 1 - fi))
                    obj_future_mask.extend([False] * (n_future - 1 - fi))
                    break
                obj_kp_future = [obj_kp_future[j][fps_idx] for j, fps_idx in enumerate(self.fps_idx_list)]
                obj_kp_future = np.concatenate(obj_kp_future, axis=0) # (N = instance_num * rand_ptcl_num, 3)
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
        p_rigid = np.ones(max_n, dtype=np.float32)  # clothes are nonrigid
        p_instance = np.zeros((max_nobj, max_n), dtype=np.float32)
        j_perm = np.random.permutation(instance_num)
        for j in range(instance_num):
            p_instance[j * ptcl_per_instance : (j + 1) * ptcl_per_instance, j_perm[j]] = 1  # TODO different number of particles per instance
        physics_param = np.zeros(max_nobj, dtype=np.float32)  # 1-dim

        # construct attributes
        attr_dim = 2
        assert attr_dim == args.attr_dim
        attrs = np.zeros((max_nobj + max_neef, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + eef_kp_num, 1] = 1.

        # construct states
        states = np.concatenate([obj_kp, eef_kp[0]], axis=0)  # (N, 3)  # the current eef_kp

        # action encoded as state_delta (only stored in eef keypoints)
        states_delta = np.zeros((max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
        states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

        # construct future states and actions
        eef_future = np.zeros((n_future, max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
        eef_future[0, max_nobj : max_nobj + eef_kp_num] = eef_kp[0]
        states_delta_future = np.zeros((n_future, max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
        states_delta_future[0, max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]
        if n_future > 1:
            eef_kp_last = eef_kp[1]
            for fi in range(n_future - 1):
                if i + fi + 1 >= len(self.obj_kypts_paths):
                    # exceed the dataset
                    continue
                eef_kp_path_future = self.eef_kypts_paths[i + fi + 1]
                eef_kp_future = np.load(eef_kp_path_future).astype(np.float32)   # (2, 8, 3)
                if obj_future_mask[fi + 1]:
                    assert np.max(eef_kp_future[0].mean(0) - eef_kp_last.mean(0)) < 0.005
                eef_future[fi + 1, max_nobj : max_nobj + eef_kp_num] = eef_kp_future[0]
                states_delta_future[fi + 1, max_nobj : max_nobj + eef_kp_num] = eef_kp_future[1] - eef_kp_future[0]
                eef_kp_last = eef_kp_future[1]

        # history state
        states = states[None]  # n_his = 1  # TODO make this code compatible with n_his > 1

        # add randomness
        # state randomness
        states += np.random.uniform(-0.005, 0.005, size=states.shape)  # TODO tune noise level
        # rotation randomness (already translation-invariant)
        random_rot = np.random.uniform(-np.pi, np.pi)
        rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                            [np.sin(random_rot), np.cos(random_rot), 0],
                            [0, 0, 1]], dtype=states.dtype)  # 2D rotation matrix in xy plane
        states = states @ rot_mat[None]
        states_delta = states_delta @ rot_mat
        eef_future = eef_future @ rot_mat[None]
        states_delta_future = states_delta_future @ rot_mat[None]
        obj_kp_futures = obj_kp_futures @ rot_mat[None]

        # save graph
        graph = {
            # input information
            "state": states,  # (n_his, N+M, state_dim)
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
            "physics_param": physics_param,  # (N,)
            "state_mask": state_mask,  # (N+M,)
            "eef_mask": eef_mask,  # (N+M,)
            "obj_mask": obj_mask,  # (N,)
        }
        # print([f"{key}: {val.shape}" for key, val in graph.items()])
        return graph
