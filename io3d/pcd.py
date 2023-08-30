import numpy as np
# import pcl
import open3d as o3d


def read_point_cloud(filename):
    # return np.asarray(pcl.load(filename))
    pcd = o3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points)


def save_point_cloud(filename, points):
    # pcl.save(pcl.PointCloud(points.astype(np.float32)),
    #          filename, format='pcd', binary=True)
    o3d.io.write_point_cloud(filename, points.astype(np.float32))
