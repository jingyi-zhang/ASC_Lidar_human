from matplotlib.pyplot import axis

import numpy as np
import pcl
import torch


def rotation(X, matrix):
    if type(X) == np.ndarray:
        return np.dot(matrix, X.T).T
    return matrix.to(X.device).matmul(X.T).T


def affine(X, matrix):
    n = X.shape[0]
    if type(X) == np.ndarray:
        res = np.concatenate((X, np.ones((n, 1))), axis=-1).T
        res = np.dot(matrix, res).T
    else:
        res = torch.cat((X, torch.ones((n, 1)).to(X.device)), axis=-1)
        res = matrix.to(X.device).matmul(res.T).T
    return res[..., :-1]


def lidar_to_camera(X, extrinsic_matrix):
    return affine(X, extrinsic_matrix)


def icp_without_rotation(source_points, target_points, target_kdtree, iteration=10):
    source_points_copy = source_points.copy()
    translation = np.zeros((3, ), dtype=np.float32)
    first = True
    while iteration:
        iteration -= 1
        if first:
            first = False
            ind = np.arange(target_points.shape[0])
        else:
            ind = target_kdtree.nearest_k_search_for_cloud(
                pcl.PointCloud(source_points_copy), 1)[0].flatten()
        offset = np.average(
            target_points[ind], axis=0) - np.average(source_points_copy, axis=0)
        translation += offset
        source_points_copy += offset
    return translation


def get_mocap_to_lidar_translation(mocap_points, lidar_points, lidar_to_mocap_rotation):
    source_points = mocap_points.astype(np.float32)
    target_points = affine(
        lidar_points, lidar_to_mocap_rotation).astype(np.float32)
    target_kdtree = pcl.PointCloud(target_points).make_kdtree_flann()
    return icp_without_rotation(source_points, target_points, target_kdtree)


def get_mocap_to_lidar_rotation(mocap_points, lidar_points, lidar_to_mocap_rotation):
    translation = get_mocap_to_lidar_translation(
        mocap_points, lidar_points, lidar_to_mocap_rotation)
    source_cloud = pcl.PointCloud(
        (mocap_points + translation).astype(np.float32))
    target_points = affine(lidar_points, lidar_to_mocap_rotation)
    target_cloud = pcl.PointCloud(target_points.astype(np.float32))
    icp = source_cloud.make_IterativeClosestPoint()
    rotation = icp.icp(source_cloud, target_cloud)[1]  # 4 * 4 ndarray
    rotation[0][3] = rotation[1][3] = rotation[2][3] = 0
    return rotation


def mocap_to_lidar(mocap_points, lidar_to_mocap_RT, lidar_points=None, translation=None):
    assert lidar_points is not None or translation is not None
    mocap_points_translated = mocap_points.astype(np.float32)
    if translation is None:
        translation = get_mocap_to_lidar_translation(
            mocap_points, lidar_points, lidar_to_mocap_RT)
    mocap_points_translated += translation
    return affine(mocap_points_translated, np.linalg.inv(lidar_to_mocap_RT))


def camera_to_pixel(X, intrinsic_matrix, distortion_coefficients):
    # focal length
    f = np.array([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]])
    # center principal point
    c = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / X[..., 2:]
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c
