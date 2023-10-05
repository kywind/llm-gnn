import glob
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import gen_args
from gnn.model_wrapper import gen_model
from gnn.utils import set_seed

from train.rigid_dataset import RigidDynDataset
import open3d as o3d

import cv2
import glob
from PIL import Image
import pickle as pkl


def train_rigid(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    data_dir = "../data/2023-09-04-18-42-27-707743"
    phases = ['train', 'val']
    batch_size = 64
    n_epoch = 2000
    datasets = {phase: RigidDynDataset(args, data_dir, phase) for phase in phases}

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=8,
    ) for phase in phases}
 
    model = gen_model(args, material_dict=None, verbose=True, debug=False)
    for epoch in range(n_epoch):
        for phase in phases:
            for i, data in enumerate(dataloaders[phase]):
                import ipdb; ipdb.set_trace()
 

if __name__ == "__main__":
    args = gen_args()
    train_rigid(args)
