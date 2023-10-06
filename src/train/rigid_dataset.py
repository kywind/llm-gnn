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
        data_dir,
        phase='train',
    ):
        self.args = args
        self.data_dir = data_dir
        self.phase = phase
        
        graph_dir = os.path.join(data_dir, 'graph')
        self.graph_paths = sorted(glob.glob(os.path.join(graph_dir, '*.pkl')))
        print(f'Found {len(self.graph_paths)} graphs in {graph_dir}')

    def __len__(self):
        return len(self.graph_paths)
    
    def __getitem__(self, idx):
        graph_path = self.graph_paths[idx]
        graph_path_next = self.graph_paths[idx + 1]
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        with open(graph_path_next, 'rb') as f:
            graph_next = pickle.load(f)
        # get movements
        graph['gt_state'] = graph_next['state']
        graph['gt_motion'] = graph_next['state'] - graph['state']
        return graph
