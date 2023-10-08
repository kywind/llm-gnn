import glob
import numpy as np
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import gen_args
from gnn.model_wrapper import gen_model
from gnn.utils import set_seed

from train.rigid_dataset import RigidDynDataset
import open3d as o3d

import cv2
import glob
from PIL import Image
import pickle as pkl


def grad_manager(phase):
    if phase == 'train':
        return torch.enable_grad()
    else:
        return torch.no_grad()
    
def parse_gt(data):
    gt_pos = data['pos'][:, 1:, :, :]
    gt_pos = gt_pos.reshape(gt_pos.shape[0], gt_pos.shape[1], -1)
    return gt_pos

def train_rigid(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    data_dirs = {
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
    out_dir = "../log/shoe_debug_2"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    phases = ['train', 'valid']
    batch_size = 64
    n_epoch = 2000
    log_interval = 10
    datasets = {phase: RigidDynDataset(args, data_dirs, phase) for phase in phases}

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=8,
    ) for phase in phases}
 
    model, loss_funcs = gen_model(args, material_dict=None, verbose=True, debug=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        for phase in phases:
            with grad_manager(phase):
                if phase == 'train': model.train()
                else: model.eval()
                loss_sum = 0
                if phase == 'valid':
                    loss_sum_list = []
                for i, data in enumerate(dataloaders[phase]):
                    if phase == 'train':
                        optimizer.zero_grad()
                    data = {key: data[key].to(device) for key in data.keys()}
                    pred_state, pred_motion = model(**data)

                    gt_state = data['state_next']
                    pred_state_p = pred_state[:, :gt_state.shape[1], :3]
                    loss = [func(pred_state_p, gt_state) for func in loss_funcs]
                    loss_sum = sum(loss)
                    if phase == 'train':
                        loss_sum.backward()
                        optimizer.step()
                        if i % log_interval == 0:
                            print(f'Epoch {epoch}, iter {i}, loss {loss_sum.item()}')
                    if phase == 'valid':
                        loss_sum_list.append(loss_sum.item())
                if phase == 'valid':
                    print(f'\nEpoch {epoch}, valid loss {np.mean(loss_sum_list)}\n')

        if (epoch < 100 and epoch % 10 == 0) or epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', f'model_{epoch}.pth'))


def test_rigid(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    data_dirs = {'test': ["../data/2023-08-23-12-08-12-201998"]}
    batch_size = 1
    save_interval = 10
    dataset = RigidDynDataset(args, data_dirs, 'test')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    out_dir = "../log/shoe_debug_2"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pred_graphs'), exist_ok=True)
 
    model, loss_funcs = gen_model(args, checkpoint=os.path.join(out_dir, 'checkpoints', 'model_100.pth'),
                                  material_dict=None, verbose=True, debug=False)
    model.to(device)

    with torch.no_grad():
        model.eval()
        loss_sum_list = []
        for i, data in enumerate(dataloader):
            data = {key: data[key].to(device) for key in data.keys()}
            pred_state, pred_motion = model(**data)

            gt_state = data['state_next']
            pred_state_p = pred_state[:, :gt_state.shape[1], :3]
            loss = [func(pred_state_p, gt_state) for func in loss_funcs]
            loss_sum = sum(loss)
            loss_sum_list.append(loss_sum.item())

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
                    'action': data['action'],
                    "p_rigid": data['p_rigid'],
                    "p_instance": data['p_instance'],
                    "Rr": data['Rr'], 
                    "Rs": data['Rs'],
                    "index": np.array([i], dtype=np.int32),
                }
                save_path = os.path.join(out_dir, 'pred_graphs', f'{i:06}.pkl')
                pkl.dump(save_graph, open(save_path, 'wb'))
            
        print(f'Test loss {np.mean(loss_sum_list)}')

if __name__ == "__main__":
    args = gen_args()
    # train_rigid(args)
    test_rigid(args)
