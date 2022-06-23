
import json
import os
import numpy as np
import open3d as o3d

def affine(X, matrix):
    assert type(X) == np.ndarray
    res = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1).T
    res = np.dot(matrix, res).T
    return res[:, :3]

def read_array_dat(filename):
    import struct
    array = []
    with open(filename, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        for i in range(n):
            array.append(struct.unpack('d', f.read(8))[0])
    return array

def prepare_dataset_dirs(dataset_prefix):
    subdir = {
        'root': ['calib', 'images', 'labels', 'pointclouds', 'mocaps', 'project'],
        'labels': ['2d', '3d'],
        '2d': ['mask', 'bbox', 'keypoints'],
        '3d': ['pose', 'segment', 'smpl', 'depth'],
        'project': ['pc', 'mocap']
    }

    dataset_dirs = {}
    import queue
    q = queue.Queue()
    q.put(('root', ''))
    while not q.empty():
        u, path = q.get()
        if u not in subdir:
            cur_path = os.path.join(dataset_prefix, path)
            dataset_dirs[u + '_dir'] = cur_path
            os.makedirs(cur_path, exist_ok=True)
        else:
            for v in subdir[u]:
                q.put((v, os.path.join(path, v)))
    return dataset_dirs

from collections import namedtuple

def prepare_current_dirs(raw_dir, dataset_dirs, index):
    cur_dirs = {'raw_dir': os.path.join(raw_dir, str(index))}
    for key, value in dataset_dirs.items():
        if key == 'calib_dir':
            cur_dirs[key] = value
        else:
            cur_dirs[key] = os.path.join(value, str(index))
        os.makedirs(cur_dirs[key], exist_ok=True)
    return dict_to_struct(cur_dirs)

def dict_to_struct(d):
    return namedtuple('Struct', d.keys())(*d.values())


def save_smpl_json(dir, pose, trans):
    beta = np.zeros((10,))
    with open(dir, 'w') as f:
        d = {'beta': beta.tolist(),
             'pose': pose.tolist(),
             'trans': trans.tolist()}
        f.write(json.dumps(d))

def read_obj(path):
    with open(path, errors='ignore') as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == 'v':
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
    return np.array(points)

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)

def ransac_registration(src_points, dst_points, voxel_size=0.05, distance_multiplier=1.5, max_iterations=1000000, confidence=0.999, mutual_filter=False):
    distance_threshold = distance_multiplier * voxel_size

    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    if isinstance(src_points, tuple):
        src_down, src_fpfh = src_points
    else:
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(src_points)
        src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)

    if isinstance(dst_points, tuple):
        dst_down, dst_fpfh = dst_points
    else:
        dst = o3d.geometry.PointCloud()
        dst.points = o3d.utility.Vector3dVector(dst_points)
        dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)

    #print('Running RANSAC')
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        mutual_filter=mutual_filter,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iterations, confidence))

    return result

def point_to_point_icp(source, target, threshold, trans_init):
    if isinstance(source, np.ndarray):
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(source)
        source = src

    if isinstance(target, np.ndarray):
        tar = o3d.geometry.PointCloud()
        tar.points = o3d.utility.Vector3dVector(target)
        target = tar

    #print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #print(reg_p2p)
    return reg_p2p


def point_to_plane_icp(source, target, threshold, trans_init):
    if isinstance(source, np.ndarray):
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(source)
        source = src

    if isinstance(target, np.ndarray):
        tar = o3d.geometry.PointCloud()
        tar.points = o3d.utility.Vector3dVector(target)
        target = tar
    target.estimate_normals()
    #print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    #print(reg_p2l)
    return reg_p2l

def fix_points_num(points: np.array, num_points: int):
    points = points[~np.isnan(points).any(axis=-1)]

    origin_num_points = points.shape[0]
    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    if origin_num_points >= num_points:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res

def pcd_from_np(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd
