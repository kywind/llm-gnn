import os
import torch
import numpy as np


img_dir = 'vis/apples_640.png'
depth_dir = 'vis/apples_640_depth.npy'

os.system(f"python MiDaS/run.py --input {img_dir} --output {depth_dir} --model MiDaS/weights/dpt_beit_large_512.pt")

