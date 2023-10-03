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
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

from data.utils import depth2fgpcd, np2o3d


class DynDatabase:
    def __init__(self, database_name, data_root):
        self.database_name = database_name
        self.root_dir = os.path.join(data_root, 'data', database_name)
        total_time = len(glob.glob(os.path.join(self.root_dir, 'camera_0/color_*.png')))
        self.image_times = list(range(total_time))

        self.camera_extrinsics = {}
        for img_id in self.image_ids:
            camera_extrinsics = np.load(os.path.join(self.root_dir, f'camera_{img_id}', 'camera_extrinsics.npy'))
            self.camera_extrinsics[img_id] = camera_extrinsics[:3,:]

        self.base_in_tag_frame = np.load(os.path.join(self.root_dir, 'base_pose_in_tag.npy'))
        self.finger_points_in_finger_frame = np.array([[0.01, -0.025, 0.014], # bottom right
                                                        [-0.01, -0.025, 0.014], # bottom left
                                                        [-0.01, -0.025, 0.051], # top left
                                                        [0.01, -0.025, 0.051]]) # top right
        self.has_push = os.path.exists(os.path.join(self.root_dir, 'push'))
        self.get_timestamps = np.loadtxt(os.path.join(self.root_dir, 'timestamp.txt')) 

    # get the grasping pose for instance_name in database_name
    def get_grasp_pose(self):
        times = self.image_times

        # grasp_pose is the pose when the distance between the fingers is the smallest
        grasp_pose = self.get_grasper_pose(times[0])
        min_dist = self.get_grasper_dist(times[0])
        for t in times:
            curr_dist = self.get_grasper_dist(t)
            if curr_dist < min_dist:
                min_dist = curr_dist
                grasp_pose = self.get_grasper_pose(t)
        
        # - 'grasp_pose': 4x4 homogeneous matrix of the grasping pose
        return grasp_pose

    def get_finger_points(self, sel_time):
        left_finger_in_base_frame = np.loadtxt(os.path.join(self.root_dir, 'pose', 'left_finger', f'{sel_time}.txt'))
        right_finger_in_base_frame = np.loadtxt(os.path.join(self.root_dir, 'pose', 'right_finger', f'{sel_time}.txt'))
        left_finger_in_tag_frame = self.base_in_tag_frame @ left_finger_in_base_frame
        right_finger_in_tag_frame = self.base_in_tag_frame @ right_finger_in_base_frame
        N = self.finger_points_in_finger_frame.shape[0]
        finger_points_in_finger_frame_cat = np.concatenate([self.finger_points_in_finger_frame, np.ones((N, 1))], axis=1) # (N, 4)
        left_finger_points = left_finger_in_tag_frame @ finger_points_in_finger_frame_cat.T # (4, N)
        right_finger_points = right_finger_in_tag_frame @ finger_points_in_finger_frame_cat.T # (4, N)
        left_finger_points = left_finger_points[:3, :].T # (N, 3)
        right_finger_points = right_finger_points[:3, :].T # (N, 3)
        return left_finger_points, right_finger_points

    def get_grasper_dist(self, sel_time):
        left_finger_points, right_finger_points = self.get_finger_points(sel_time)
        return np.linalg.norm(left_finger_points - right_finger_points, axis=1).mean()

    def get_grasper_pose(self, sel_time):
        robotiq_pose_in_base_frame = np.loadtxt(os.path.join(self.root_dir, 'pose', 'robotiq_base', f'{sel_time}.txt'))
        robotiq_pose_in_tag_frame = self.base_in_tag_frame @ robotiq_pose_in_base_frame
        return robotiq_pose_in_tag_frame

    def get_pushes(self):
        assert self.has_push
        push_num = len(os.listdir(os.path.join(self.root_dir, 'push')))
        push_pts = [np.load(os.path.join(self.root_dir, 'push', f'{i}', 'push_pts.npy')) for i in range(push_num)]
        push_time = [np.load(os.path.join(self.root_dir, 'push', f'{i}', 'push_time.npy')) for i in range(push_num)]
        push_pts = np.stack(push_pts, axis=0) # (push_num, 2, 3)
        push_time = np.stack(push_time, axis=0) # (push_num, 2)
        return push_pts, push_time

