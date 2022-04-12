import numpy as np
import pcl


def read_point_cloud(filename):
    return np.asarray(pcl.load(filename))


def save_point_cloud(filename, points):
    pcl.save(pcl.PointCloud(points.astype(np.float32)),
             filename, format='pcd', binary=True)
