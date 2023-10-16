import glob
import numpy as np
import os
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

from train.random_rigid_dataset import RandomRigidDynDataset
from train.multistep_rigid_dataset import MultistepRigidDynDataset, construct_edges_from_states
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

def construct_relations(states, state_mask, eef_mask, adj_thresh_range=[0.1, 0.2], max_nR=500):
    # construct relations (density as hyperparameter)
    bsz = states.shape[0]  # states: B, n_his, N, 3
    adj_thresh = np.random.uniform(*adj_thresh_range, (bsz,))
    adj_thresh = torch.tensor(adj_thresh).to(states.device)
    Rr, Rs = construct_edges_from_states(states[:, -1], adj_thresh, 
                                        mask=state_mask, 
                                        eef_mask=eef_mask,
                                        no_self_edge=True)
    Rr = Rr.detach()
    Rs = Rs.detach()
    return Rr, Rs

def train_rigid(args, out_dir, data_dirs, dense=True, material='rigid', ratios=None):
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
        ratios = {"train": [0, 1], "valid": [0, 1]}
    batch_size = 64
    n_epoch = 500
    log_interval = 5
    dist_thresh = 0.02
    n_future = 3
    datasets = {phase: MultistepRigidDynDataset(args, data_dirs, prep_save_dir, ratios, phase, dense, 
                fixed_idx=False, dist_thresh=dist_thresh, n_future=n_future) for phase in phases}

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
        for phase in phases:
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
                    future_eef = data['eef_future']  # (B, n_future, n_p+n_s, 3)
                    future_action = data['action_future']  # (B, n_future, n_p+n_s, 3)
                    assert torch.sum(future_action[:, 0] - data['action']) < 1e-6

                    for fi in range(n_future):
                        gt_state = future_state[:, fi].clone()  # (B, n_p, 3)
                        gt_mask = future_mask[:, fi].clone()  # (B,)
                        # gt_mask = gt_mask[:, None, None].repeat(1, gt_state.shape[1], 3)  # (B, n_p, 3)

                        data['Rr'], data['Rs'] = construct_relations(data['state'], 
                                                                    data['state_mask'], data['eef_mask'], 
                                                                    adj_thresh_range=[0.1, 0.2])

                        pred_state, pred_motion = model(**data)

                        pred_state_p = pred_state[:, :gt_state.shape[1], :3].clone()
                        loss = [func(pred_state_p[gt_mask], gt_state[gt_mask]) for func in loss_funcs]
                        
                        if args.use_rigid_loss:
                            p_instance = data['p_instance']  # B, n_p, n_ins
                            # p_rigid = data['p_rigid']  # B, n_ins  # assume all are rigid objects
                            # p_rigid_per_particle = torch.sum(p_instance * p_rigid[:, None, :], 2, keepdim=True)  # B, n_p, 1

                            pred_state_p_valid = pred_state_p[gt_mask]  # B', n_p, 3
                            p_instance_valid = p_instance[gt_mask]  # B', n_p, n_ins
                            
                            instance_pred_state_p = p_instance_valid.transpose(1, 2)[..., None] * pred_state_p_valid[:, None, :]  # B', n_ins, n_p, 3
                            rigid_instance_pred_state_p = instance_pred_state_p[p_instance_valid.sum(1) > 0]  # n_rigid, n_p, 3
                            rigid_instance_mask = p_instance_valid.transpose(1, 2)[p_instance_valid.sum(1) > 0].bool()  # n_rigid, n_p

                            # don't use gt state, use input state instead
                            input_state_p = data['state'][:, -1, :gt_state.shape[1]]  # B, n_p, 3
                            input_state_p_valid = input_state_p[gt_mask]  # B', n_p, 3
                            instance_input_state_p = p_instance_valid.transpose(1, 2)[..., None] * input_state_p_valid[:, None, :]  # B', n_ins, n_p, 3
                            rigid_instance_input_state_p = instance_input_state_p[p_instance_valid.sum(1) > 0]  # n_rigid, n_p, 3
                            
                            # calculate rigid loss
                            loss.append(rigid_loss(rigid_instance_input_state_p, rigid_instance_pred_state_p, rigid_instance_mask))

                        loss_sum += sum(loss)

                        if fi < n_future - 1:
                            # build next graph
                            next_eef = future_eef[:, fi+1].clone()  # (B, n_p+n_s, 3)
                            next_action = future_action[:, fi+1].clone()  # (B, n_p+n_s, 3)
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
                                "physics_param": data["physics_param"],  # (B, N,)
                                "state_mask": data["state_mask"],  # (B, N+M,)
                                "eef_mask": data["eef_mask"],  # (B, N+M,)
                                "obj_mask": data["obj_mask"],  # (B, N,)
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
                if phase == 'valid':
                    print(f'\nEpoch {epoch}, valid loss {np.mean(loss_sum_list)}\n')

                if phase == 'train':
                    loss_plot_list_train.append(np.mean(loss_sum_list))
                if phase == 'valid':
                    loss_plot_list_valid.append(np.mean(loss_sum_list))
                
        if ((epoch + 1) < 100 and (epoch + 1) % 10 == 0) or (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', f'model_{(epoch + 1)}.pth'))
    
            # save figures
            plt.figure(figsize=(20, 5))
            plt.plot(loss_plot_list_train, label='train')
            plt.plot(loss_plot_list_valid, label='valid')
            # ax = plt.gca()
            # y_min = min(min(loss_plot_list_train), min(loss_plot_list_valid))
            # y_min = min(loss_plot_list_valid)
            # y_max = min(5 * y_min, max(max(loss_plot_list_train), max(loss_plot_list_valid)))
            # ax.set_ylim([0, y_max])
            plt.legend()
            plt.savefig(os.path.join(out_dir, 'loss.png'), dpi=300)
            plt.close()


def test_rigid(args, out_dir, data_dirs, checkpoint, dense=True, material='rigid', ratios=None):
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
    dataset = RandomRigidDynDataset(args, data_dirs, ratios, phase='test', dense=dense)

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

    # out_dir = "../log/rigid_dense_debug_1"
    # out_dir = "../log/rigid_dense_debug_2"  # pstep = 6
    out_dir = "../log/rigid_dense_debug_3"  # pstep = 6, output is 3 dim
    dense = True
    train_data_dirs = {
        "train": [
            "../data/2023-08-30-03-04-49-509758",
            "../data/2023-08-30-02-02-53-617979",
            "../data/2023-08-30-01-45-16-257548",
            "../data/2023-08-23-12-23-07-775716",
            "../data/2023-08-30-02-20-39-572700",
            "../data/2023-08-30-03-44-14-098121",
            "../data/2023-08-30-00-54-02-790828",
            "../data/2023-08-30-02-48-27-532912",
            "../data/2023-08-29-17-49-04-904390",
            "../data/2023-08-30-03-27-55-702301",
            "../data/2023-08-29-18-07-15-165315",
            "../data/2023-08-23-12-17-53-370195",
            "../data/2023-08-30-04-16-04-497588",
            # "../data/2023-08-29-21-23-47-258600",
            "../data/2023-08-30-04-00-39-286249",
            "../data/2023-08-29-20-10-13-123194",
            "../data/2023-09-04-18-26-45-932029",
            "../data/2023-08-30-00-47-16-839238",
        ],
        "valid": [
            "../data/2023-08-23-12-08-12-201998",
            "../data/2023-09-04-18-42-27-707743",
        ],
    }
    train_rigid(args, out_dir, train_data_dirs, dense)

    # test_data_dirs = {
    #     "test": [
    #         "../data/2023-08-23-12-08-12-201998",
    #         "../data/2023-09-04-18-42-27-707743",
    #     ],
    # }  # not in training set
    test_data_dirs = "../data/2023-08-23-12-08-12-201998"  # not in training set
    # test_data_dirs = "../data/2023-09-04-18-42-27-707743"  # not in training set
    # test_data_dirs = "../data/2023-08-23-12-23-07-775716"  # in training set
    checkpoint = "model_2000.pth"
    # test_rigid(args, out_dir, test_data_dirs, checkpoint, dense)
