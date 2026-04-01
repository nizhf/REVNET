import torch
import numpy as np
import open3d as o3d


def read_point_cloud(path):
    pcd = o3d.io.read_point_cloud(str(path))
    pcd = np.array(pcd.points, dtype=np.float32)
    pcd = torch.from_numpy(pcd)
    return pcd
