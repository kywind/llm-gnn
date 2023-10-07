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

    
def parse_gt(data):
    gt_pos = data['pos'][:, 1:, :, :]
    gt_pos = gt_pos.reshape(gt_pos.shape[0], gt_pos.shape[1], -1)
    return gt_pos

def train_rigid(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    data_dir = "../data/2023-09-04-18-42-27-707743"
    phases = ['train']
    batch_size = 64
    n_epoch = 2000
    log_interval = 10
    datasets = {phase: RigidDynDataset(args, data_dir, phase) for phase in phases}

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=8,
    ) for phase in phases}
 
    model, loss_funcs = gen_model(args, material_dict=None, verbose=True, debug=False)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        for phase in phases:
            for i, data in enumerate(dataloaders[phase]):
                optimizer.zero_grad()
                data = {key: data[key].to(device) for key in data.keys()}
                pred_state, pred_motion = model(**data)

                gt_state = data['state_next']
                pred_state_p = pred_state[:, :gt_state.shape[1], :3]
                loss = [func(pred_state_p, gt_state) for func in loss_funcs]
                loss_sum = sum(loss)
                loss_sum.backward()
                optimizer.step()
                if i % log_interval == 0:
                    print(f'Epoch {epoch}, iter {i}, loss {loss_sum.item()}')


if __name__ == "__main__":
    args = gen_args()
    train_rigid(args)
