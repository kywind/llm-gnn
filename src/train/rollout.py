import glob
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import gen_args
from gnn.model_wrapper import gen_model
from gnn.utils import set_seed

import open3d as o3d

import cv2
import glob
from PIL import Image
import pickle as pkl


def rollout_rigid(args):
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    data_dir = "../data/2023-08-23-12-08-12-201998"
    graph_dir = os.path.join(data_dir, 'graph')
    graph_path_list = sorted(glob.glob(os.path.join(graph_dir, '*.pkl')))
    
    start_idx = 0
    rollout_steps = 10
    with open(graph_path_list[start_idx], 'rb') as f:
        graph = pkl.load(f)
    graph = {key: graph[key].to(device) for key in graph.keys()}
    
    model, loss_funcs = gen_model(args, checkpoint='../log/shoe_debug/model_50.pth', 
                                  material_dict=None, verbose=True, debug=False)
    model = model.to(device)
    model.eval()
    gt_state_list = []
    pred_state_list = []
    for i in range(start_idx + 1, start_idx + 1 + rollout_steps):
        with open(graph_path_list[start_idx], 'rb') as f:
            gt_graph = pkl.load(f)
        gt_state = gt_graph['state_next'].detach().cpu().numpy()
        gt_state_list.append(gt_graph)
        pred_state, pred_motion = model(**graph)
        pred_state = pred_state.detach().cpu().numpy()


if __name__ == "__main__":
    args = gen_args()
    rollout_rigid(args)
