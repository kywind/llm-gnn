import os
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl
import copy


# shoe pushing
data_dir = "../data/2023-09-04-18-42-27-707743"
save_dir = f"vis/graph-vis-raw-{data_dir.split('/')[-1]}-dense"
os.makedirs(save_dir, exist_ok=True)

t_list = np.loadtxt(os.path.join(data_dir, 'push_t_list_2_dense.txt'), dtype=np.int32)


intr_list = [None] * 4
extr_list = [None] * 4
for cam in range(4):
    save_dir_cam = os.path.join(save_dir, f'camera_{cam}')
    os.makedirs(save_dir_cam, exist_ok=True)
    intr_list[cam] = np.load(os.path.join(data_dir, f"camera_{cam}/camera_params.npy"))
    extr_list[cam] = np.load(os.path.join(data_dir, f"camera_{cam}/camera_extrinsics.npy"))

i_min = 0
i_max = 100
for i in range(i_min, i_max):

    push_start = t_list[i, 0]
    push_end = t_list[i, 1]

    for cam in range(4):
        for t in range(push_start, push_end):
            img = cv2.imread(os.path.join(data_dir, f"camera_{cam}/color/color_{t}.png"))
            print(os.path.join(save_dir, f'camera_{cam}', f'{t:06}.png'))
            cv2.imwrite(os.path.join(save_dir, f'camera_{cam}', f'{t:06}.png'), img)
