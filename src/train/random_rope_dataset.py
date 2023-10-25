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
from dgl.geometry import farthest_point_sampler



def construct_edges_from_states(states, adj_thresh, mask, eef_mask, no_self_edge=False):  # helper function for construct_graph
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: float
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
    adj_matrix = ((dis - threshold) < 0).float()

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
class RandomRopeDynDataset(Dataset):
    def __init__(self, 
        args,
        data_dirs,
        ratios,
        phase='train',
        dense=True,
        fixed_idx=False,
    ):
        self.args = args
        self.phase = phase
        if isinstance(data_dirs, dict):
            self.data_dirs = data_dirs[phase]
        elif isinstance(data_dirs, str):  # single data directory
            self.data_dirs = [data_dirs]
        print(f'Setting up {phase} dataset')
        print(f'Found {len(self.data_dirs)} data directories')

        self.obj_kypts_paths = []
        self.eef_kypts_paths = []
        self.physics_paths = []
        for data_dir in self.data_dirs:
            eef_kypts_dir = os.path.join(data_dir, 'kypts', 'eef_kypts')
            eef_kypts_path_list = sorted(glob.glob(os.path.join(eef_kypts_dir, '*.npy')))
            obj_kypts_dir = os.path.join(data_dir, 'kypts', 'obj_kypts')
            obj_kypts_path_list = sorted(glob.glob(os.path.join(obj_kypts_dir, '*.npy')))
            physics_dir = os.path.join(data_dir, 'kypts', 'physics')
            physics_path_list = sorted(glob.glob(os.path.join(physics_dir, '*.npy')))
            self.obj_kypts_paths.extend(obj_kypts_path_list)
            self.eef_kypts_paths.extend(eef_kypts_path_list)
            self.physics_paths.extend(physics_path_list)
            print(f'Found {len(obj_kypts_path_list)} obj keypoint files in {obj_kypts_dir}')
            print(f'Found {len(eef_kypts_path_list)} eef keypoint files in {eef_kypts_dir}')
            assert len(obj_kypts_path_list) == len(eef_kypts_path_list)
        print(f'Found {len(self.obj_kypts_paths)} obj keypoint files in total')
        print(f'Found {len(self.eef_kypts_paths)} eef keypoint files in total')
        assert len(self.obj_kypts_paths) == len(self.eef_kypts_paths)

        self.ratio = ratios[phase]
        print(f'Taking ratio {self.ratio} of {len(self.eef_kypts_paths)} eef keypoint files')
        self.eef_kypts_paths = self.eef_kypts_paths[int(len(self.eef_kypts_paths) * self.ratio[0]):int(len(self.eef_kypts_paths) * self.ratio[1])]
        self.obj_kypts_paths = self.obj_kypts_paths[int(len(self.obj_kypts_paths) * self.ratio[0]):int(len(self.obj_kypts_paths) * self.ratio[1])]
        print(f'{phase} dataset has {len(self.eef_kypts_paths)} eef keypoint files and {len(self.obj_kypts_paths)} obj keypoint files')

        self.max_n = 2  # max number of objects
        self.max_nobj = 40  # max number of object points
        self.max_neef = 8  # max number of eef points
        self.max_nR = 500  # max number of relations

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
        max_nR = self.max_nR
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

        # TODO construct relations here or after collate_fn?
        # construct relations (density as hyperparameter)
        adj_thresh = np.random.uniform(*self.adj_thresh_range)
        states = np.concatenate([obj_kp, eef_kp[0]], axis=0)  # (N, 3)  # the current eef_kp
        Rr, Rs = construct_edges_from_states(torch.tensor(states).unsqueeze(0), adj_thresh, 
                                            mask=torch.tensor(state_mask).unsqueeze(0), 
                                            eef_mask=torch.tensor(eef_mask).unsqueeze(0),
                                            no_self_edge=True)
        Rr, Rs = Rr.squeeze(0).numpy(), Rs.squeeze(0).numpy()

        # action encoded as state_delta (only stored in eef keypoints)
        states_delta = np.zeros((max_nobj + max_neef, states.shape[-1]), dtype=np.float32)
        states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp[1] - eef_kp[0]

        # history state
        states = states[None]  # n_his = 1  # TODO make this code compatible with n_his > 1

        if max_nR is not None:
            Rr = np.pad(Rr, ((0, max_nR - Rr.shape[0]), (0, 0)), mode='constant')
            Rs = np.pad(Rs, ((0, max_nR - Rs.shape[0]), (0, 0)), mode='constant')
        else:
            raise Exception("not using max_nR which only works for fixed number of relations")

        # save graph
        graph = {
            "attrs": attrs,  # (N+M, attr_dim)
            "state": states,  # (n_his, N+M, state_dim)
            "action": states_delta,  # (N+M, state_dim)
            "Rr": Rr,  # (n_rel, N+M)
            "Rs": Rs,  # (n_rel, N+M)
            "p_rigid": p_rigid,  # (n_instance,)
            "p_instance": p_instance,  # (N, n_instance)
            "physics_param": physics_param,  # (N,)
            "state_next": obj_kp_next,  # (N, state_dim)
            "obj_mask": obj_mask,  # (N,)
        }
        # print([f"{key}: {val.shape}" for key, val in graph.items()])
        return graph
