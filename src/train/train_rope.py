import glob
import numpy as np
import os
import time
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import gen_args
from gnn.model_wrapper import gen_model
from gnn.utils import set_seed, umeyama_algorithm

from train.random_rope_dataset import RandomRopeDynDataset
from train.multistep_rope_dataset import MultistepRopeDynDataset
from train.canonical_rope_dataset import CanonicalRopeDynDataset, construct_edges_from_states
from train.linear_rope_dataset import LinearRopeDynDataset, construct_edges_from_adjacency
import open3d as o3d

import cv2
import glob
from PIL import Image
import pickle as pkl
import matplotlib.pyplot as plt


def rigid_loss(orig_pos, pred_pos, obj_mask):
    _, R_pred, t_pred = umeyama_algorithm(orig_pos, pred_pos, obj_mask, fixed_scale=True)
    pred_pos_ume = orig_pos.bmm(R_pred.transpose(1, 2)) + t_pred
    pred_pos_ume = pred_pos_ume.detach()
    loss = F.mse_loss(pred_pos[obj_mask], pred_pos_ume[obj_mask])
    return loss

def grad_manager(phase):
    if phase == 'train':
        return torch.enable_grad()
    else:
        return torch.no_grad()

def truncate_graph(data):
    Rr = data['Rr']
    Rs = data['Rs']
    Rr_nonzero = torch.sum(Rr, dim=-1) > 0
    Rs_nonzero = torch.sum(Rs, dim=-1) > 0
    n_Rr = torch.max(Rr_nonzero.sum(1), dim=0)[0].item()
    n_Rs = torch.max(Rs_nonzero.sum(1), dim=0)[0].item()
    max_n = max(n_Rr, n_Rs)
    data['Rr'] = data['Rr'][:, :max_n, :]
    data['Rs'] = data['Rs'][:, :max_n, :]
    return data

def construct_relations(states, state_mask, eef_mask, adj_thresh_range=[0.1, 0.2], max_nR=500, adjacency=None):
    # construct relations (density as hyperparameter)
    bsz = states.shape[0]  # states: B, n_his, N, 3
    adj_thresh = np.random.uniform(*adj_thresh_range, (bsz,))
    adj_thresh = torch.tensor(adj_thresh).to(states.device)
    Rr, Rs = construct_edges_from_adjacency(states[:, -1], adj_thresh, 
                                        mask=state_mask, 
                                        eef_mask=eef_mask,
                                        adjacency=adjacency,
                                        no_self_edge=True)
    assert Rr[:, -1].sum() > 0
    Rr = Rr.detach()
    Rs = Rs.detach()
    return Rr, Rs

def train_rope(args, out_dir, data_dirs, dense=True, material='rope', ratios=None):
    torch.autograd.set_detect_anomaly(True)
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    prep_save_dir = os.path.join(out_dir, 'preprocess')
    os.makedirs(prep_save_dir, exist_ok=True)
    phases = ['train', 'valid']
    if ratios is None:
        raise ValueError
        ratios = {"train": [0, 1], "valid": [0, 1]}
    batch_size = 256
    n_epoch = 1000
    log_interval = 5
    n_future = 3
    dist_thresh_range = [0.02, 0.05]  # for preprocessing
    adj_thresh_range = [0.15, 0.25]  # for constructing relations
    # datasets = {phase: MultistepRopeDynDataset(args, data_dirs, prep_save_dir, ratios, phase, dense, 
    #             fixed_idx=False, dist_thresh=0.05, n_future=n_future) for phase in phases}
    # datasets = {phase: CanonicalRopeDynDataset(args, data_dirs, prep_save_dir, ratios, phase, 
    #             fixed_idx=False, dist_thresh_range=dist_thresh_range, n_future=n_future) for phase in phases}
    datasets = {phase: LinearRopeDynDataset(args, data_dirs, prep_save_dir, ratios, phase, 
                fixed_idx=False, dist_thresh_range=dist_thresh_range, n_future=n_future) for phase in phases}

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=8,
    ) for phase in phases}
 
    model, loss_funcs = gen_model(args, material_dict=None, material=material, verbose=True, debug=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_plot_list_train = []
    loss_plot_list_valid = [] 
    for epoch in range(n_epoch):
        time1 = time.time()
        for phase in phases:
            datasets[phase].reset_epoch()
            with grad_manager(phase):
                if phase == 'train': model.train()
                else: model.eval()
                loss_sum_list = []
                for i, data in enumerate(dataloaders[phase]):
                    if phase == 'train':
                        optimizer.zero_grad()
                    data = {key: data[key].to(device) for key in data.keys()}
                    loss_sum = 0

                    future_state = data['state_future']  # (B, n_future, n_p, 3)
                    future_mask = data['state_future_mask']  # (B, n_future)
                    future_eef = data['eef_future']  # (B, n_future-1, n_p+n_s, 3)
                    future_action = data['action_future']  # (B, n_future-1, n_p+n_s, 3)

                    for fi in range(n_future):
                        gt_state = future_state[:, fi].clone()  # (B, n_p, 3)
                        gt_mask = future_mask[:, fi].clone()  # (B,)
                        # gt_mask = gt_mask[:, None, None].repeat(1, gt_state.shape[1], 3)  # (B, n_p, 3)

                        data['Rr'], data['Rs'] = construct_relations(data['state'], 
                                                                    data['state_mask'], data['eef_mask'], 
                                                                    adj_thresh_range=adj_thresh_range,
                                                                    adjacency=data['adjacency'])

                        pred_state, pred_motion = model(**data)

                        pred_state_p = pred_state[:, :gt_state.shape[1], :3].clone()
                        loss = [func(pred_state_p[gt_mask], gt_state[gt_mask]) for func in loss_funcs]

                        loss_sum += sum(loss)

                        if fi < n_future - 1:
                            # build next graph
                            next_eef = future_eef[:, fi].clone()  # (B, n_p+n_s, 3)
                            next_action = future_action[:, fi].clone()  # (B, n_p+n_s, 3)
                            next_state = next_eef.unsqueeze(1)  # (B, 1, n_p+n_s, 3)
                            next_state[:, -1, :pred_state_p.shape[1]] = pred_state_p  # TODO n_his > 1
                            next_graph = {
                                # input information
                                "state": next_state,  # (B, n_his, N+M, state_dim)
                                "action": next_action,  # (B, N+M, state_dim) 
                                
                                # attr information
                                "attrs": data["attrs"],  # (B, N+M, attr_dim)
                                "p_rigid": data["p_rigid"],  # (B, n_instance,)
                                "p_instance": data["p_instance"],  # (B, N, n_instance)
                                "physics_param": data["physics_param"],  # (B, N, phys_dim)
                                "state_mask": data["state_mask"],  # (B, N+M,)
                                "eef_mask": data["eef_mask"],  # (B, N+M,)
                                "obj_mask": data["obj_mask"],  # (B, N,)
                                "adjacency": data["adjacency"],  # (B, M',)
                            }
                            data = next_graph

                    if phase == 'train':
                        loss_sum.backward()
                        optimizer.step()
                        if i % log_interval == 0:
                            print(f'Epoch {epoch}, iter {i}, loss {loss_sum.item()}')
                            loss_sum_list.append(loss_sum.item())
                    if phase == 'valid':
                        loss_sum_list.append(loss_sum.item())
                        # if i % log_interval == 0:
                        #     print(f'[Valid] Epoch {epoch}, iter {i}, loss {loss_sum.item()}')
                        #     loss_sum_list.append(loss_sum.item())
                if phase == 'valid':
                    print(f'\nEpoch {epoch}, valid loss {np.mean(loss_sum_list)}\n')

                if phase == 'train':
                    loss_plot_list_train.append(np.mean(loss_sum_list))
                if phase == 'valid':
                    loss_plot_list_valid.append(np.mean(loss_sum_list))
        
        time2 = time.time()
        print(f'Epoch {epoch} time: {time2 - time1}')
        if ((epoch + 1) < 200 and (epoch + 1) % 10 == 0) or (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', f'model_{(epoch + 1)}.pth'))
    
        # plot figures
        plt.figure(figsize=(20, 5))
        plt.plot(loss_plot_list_train, label='train')
        plt.plot(loss_plot_list_valid, label='valid')
        # cut off figure
        ax = plt.gca()
        y_min = min(min(loss_plot_list_train), min(loss_plot_list_valid))
        y_min = min(loss_plot_list_valid)
        y_max = min(3 * y_min, max(max(loss_plot_list_train), max(loss_plot_list_valid)))
        ax.set_ylim([0, y_max])
        # save
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'loss.png'), dpi=300)
        plt.close()


def test_rope(args, out_dir, data_dirs, checkpoint, dense=True, material='rope', ratios=None):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # data_dirs = "../data/2023-08-23-12-08-12-201998"  # not in training set
    # data_dirs = "../data/2023-08-23-12-23-07-775716"  # in training set
    batch_size = 1
    save_interval = 1
    # dense = False  # shoe_debug_3 and shoe_debug_4: False
    if ratios is None:
        ratios = {"train": [0, 1], "valid": [0, 1], "test": [0, 1]}
    dataset = RandomRopeDynDataset(args, data_dirs, ratios, phase='test', dense=dense)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    # TODO this does not work with postprocess.py
    # if isinstance(data_dirs, dict):
    #     data_dirs_test = data_dirs['test']
    #     dd_name_list = []
    #     for dd in data_dirs_test:
    #         dd_name = dd.split('/')[-1]
    #         dd_name_list.append(dd_name)
    #     dd_names = '_'.join(dd_name_list)
    # else:
    #     dd_names = data_dirs.split('/')[-1]
    dd_names = data_dirs.split('/')[-1]
    pred_graph_out_dir = os.path.join(out_dir, f"pred_graphs_{dd_names}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_graph_out_dir, exist_ok=True)
 
    model, loss_funcs = gen_model(args, checkpoint=os.path.join(out_dir, 'checkpoints', checkpoint),
                                  material_dict=None, material=material, verbose=True, debug=False)
    model.to(device)

    with torch.no_grad():
        model.eval()
        loss_sum_list = []
        trivial_loss_sum_list = []
        for i, data in enumerate(dataloader):
            data = {key: data[key].to(device) for key in data.keys()}
            pred_state, pred_motion = model(**data)

            gt_state = data['state_next']
            obj_mask = data['obj_mask']
            pred_state_p = pred_state[:, :gt_state.shape[1], :3]
            loss = [func(pred_state_p[obj_mask], gt_state[obj_mask]) for func in loss_funcs]
            loss_sum = sum(loss)
            loss_sum_list.append(loss_sum.item())
            pred_state_p_trivial = data['state'][:, -1, :gt_state.shape[1], :3]
            loss_trivial = [func(pred_state_p_trivial[obj_mask], gt_state[obj_mask]) for func in loss_funcs]
            loss_trivial_sum = sum(loss_trivial)
            trivial_loss_sum_list.append(loss_trivial_sum.item())

            if i % save_interval == 0:
                data = {key: data[key].detach().cpu().numpy()[0] for key in data.keys()}
                pred_state_p = pred_state_p.detach().cpu().numpy()[0]
                gt_state = gt_state.detach().cpu().numpy()[0]
                orig_state_p = data['state'][-1, :gt_state.shape[0], :3]  # -1: removing history state
                orig_state_s = data['state'][-1, gt_state.shape[0]:, :3]
                save_graph = {
                    'orig_state_p': orig_state_p,
                    'orig_state_s': orig_state_s,
                    'pred_state_p': pred_state_p,
                    'gt_state_p': gt_state,
                    'attrs': data['attrs'],
                    'action': data['action'],
                    "p_rigid": data['p_rigid'],
                    "p_instance": data['p_instance'],
                    "Rr": data['Rr'], 
                    "Rs": data['Rs'],
                    "index": np.array([i], dtype=np.int32),
                }
                save_path = os.path.join(pred_graph_out_dir, f'{i:06}.pkl')
                pkl.dump(save_graph, open(save_path, 'wb'))
            
        print(f'Test loss {np.mean(loss_sum_list)}')
        print(f'Trivial loss {np.mean(trivial_loss_sum_list)}')


if __name__ == "__main__":
    args = gen_args()

    # out_dir = "../log/rope_can_debug_connected_pstep12"
    out_dir = "../log/rope_noabspos_can_linear_pstep12"
    dense = True  # deprecated
    train_data_dirs = "../data/rope-new"
    ratios = {"train": [0, 0.9], "valid": [0.9, 1]}
    train_rope(args, out_dir, train_data_dirs, dense, ratios=ratios)

