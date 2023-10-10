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

from data.utils import depth2fgpcd, np2o3d


class RigidDynDataset(Dataset):
    def __init__(self, 
        args,
        data_dirs,
        ratios,
        phase='train',
        dense=True,
    ):
        self.args = args
        self.phase = phase
        if isinstance(data_dirs, dict):
            self.data_dirs = data_dirs[phase]
        elif isinstance(data_dirs, str):  # single data directory
            self.data_dirs = [data_dirs]
        print(f'Setting up {phase} dataset')
        print(f'Found {len(self.data_dirs)} data directories')

        self.graph_paths = []
        for data_dir in self.data_dirs:
            graph_dir = os.path.join(data_dir, 'dense_graph' if dense else 'graph')
            graph_path_list = sorted(glob.glob(os.path.join(graph_dir, '*.pkl')))
            self.graph_paths.extend(graph_path_list)
            print(f'Found {len(graph_path_list)} graphs in {graph_dir}')
        print(f'Found {len(self.graph_paths)} graphs in total')

        self.ratio = ratios[phase]
        print(f'Taking ratio {self.ratio} of {len(self.graph_paths)} graphs')
        self.graph_paths = self.graph_paths[int(len(self.graph_paths) * self.ratio[0]):int(len(self.graph_paths) * self.ratio[1])]
        print(f'{phase} dataset has {len(self.graph_paths)} graphs')

    def __len__(self):
        return len(self.graph_paths)
    
    def __getitem__(self, idx):
        graph_path = self.graph_paths[idx]
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        return graph
