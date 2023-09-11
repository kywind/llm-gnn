import numpy as np
import torch

class PointCloud:
    def __init__(self, points, normals, colors, labels, physics_params=None):
        self.points = points
        self.normals = normals
        self.colors = colors
        self.labels = labels

        self.physics_params = physics_params
    
    def __len__(self):
        return len(self.points)

    def subsample(self, n):
        pass
