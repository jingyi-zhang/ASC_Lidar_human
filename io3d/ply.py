import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
import pcd
from plyfile import PlyData, PlyElement


# use pcl to write ply has bug
def save_point_cloud(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2])
              for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(filename)


def pcd_to_ply(pcd_filename, ply_filename):
    save_point_cloud(ply_filename, pcd.read_point_cloud(pcd_filename))


def read_point_cloud(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points
