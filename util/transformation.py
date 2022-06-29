import math

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

# 使用模拟退火算法；只进行x,y轴配准；最大化点云中位于smpl mesh表面点的数量；
def mqh_sa_icp_wo_r(smpl_v, pc, init_trans, trans_move_limit=0.2, dist_threshold=0.01, T=1, Tmin=0.5, k=5, z=0):
    pc = pc.astype(np.float32)
    kdtree = pcl.PointCloud(smpl_v.astype(np.float32)).make_kdtree_flann()

    def objective_func(t):
        ind, dist = kdtree.nearest_k_search_for_cloud(pcl.PointCloud(pc + t), 1)
        mask = dist < dist_threshold
        return mask.sum() / 10, mask

    #init_trans = smpl_v.mean(axis=0) - pc.mean(axis=0)
    trans = init_trans.astype(np.float32).copy()
    trans[2] = z
    o, m = objective_func(trans)
    #print(f'mqh_sa_icp_wo_r, init_o:{o}, trans_limit:{trans_move_limit}, T:{T}, Tmin:{Tmin}, K:{k}')
    while T >= Tmin:
        for i in range(k):
            #new_trans = trans + (np.random.rand(3) - 0.5) * np.array([1, 1, 0.5]) * T
            new_trans = trans + (np.random.rand(3) - 0.5) * 2 * trans_move_limit * T
            new_trans[2] = z
            new_trans = new_trans.astype(np.float32)

            if np.linalg.norm(init_trans-new_trans) < trans_move_limit:    #此处本来是用于判定生成的new_trans是否合理的
                new_o, new_m = objective_func(new_trans)
                if new_o - o > 0:
                    #print(f'{o:.3f}->{new_o:.3f} accept')
                    trans, o, m = new_trans, new_o, new_m
                else:
                    p = math.exp((new_o - o) / T)
                    r = np.random.uniform(low=0, high=1)
                    if r < p:
                        #print(f'{o:.3f}->{new_o:.3f} accept by p:{p:.3f}')
                        trans, o, m = new_trans, new_o, new_m
                    else:
                        #print(f'{o:.3f}->{new_o:.3f} reject by p:{p:.3f}')
                        pass
                #print(T, i, trans, o)

        T *= 0.6  # 降温函数，也可使用T=0.9T
        #print(f'fitness:{o:.3f}, T:{T:.3f}')
    return trans, m.flatten()

# 只进行x,y轴配准；最大化点云中位于smpl mesh表面点的数量；
def mqh_icp_wo_r(smpl_v, pc, init_trans, trans_move_limit=0.2, dist_threshold=0.01, T=1, Tmin=0.1, k=5, z=0):
    pc = pc.astype(np.float32)
    smpl_v = smpl_v.astype(np.float32)
    kdtree = pcl.PointCloud(smpl_v).make_kdtree_flann()
    pc_center = np.average(pc, axis=0)

    #TODO: pcl.PointCloud 应用trans
    def objective_func(t):
        ind, dist = kdtree.nearest_k_search_for_cloud(pcl.PointCloud(pc + t), 1)
        return (dist < dist_threshold).sum()


    #init_trans = smpl_v.mean(axis=0) - pc.mean(axis=0)
    trans = init_trans.copy()
    trans[2] = z

    o = objective_func(trans)

    while T >= Tmin:
        for i in range(k):
            #new_trans = trans + (np.random.rand(3) - 0.5) * np.array([1, 1, 0.5]) * T
            noice_trans = trans + (np.random.rand(3) - 0.5) * 2 * trans_move_limit * T
            noice_trans[2] = z
            noice_trans = noice_trans.astype(np.float32)

            ind, dist = kdtree.nearest_k_search_for_cloud(pcl.PointCloud(pc + noice_trans), 1)
            #o = (dist < dist_threshold).sum()

            new_trans = np.average(smpl_v[ind.flatten()], axis=0) - pc_center
            new_trans[2] = z

            if np.linalg.norm(init_trans-new_trans) < trans_move_limit:    #此处本来是用于判定生成的new_trans是否合理的
                new_o = objective_func(new_trans)
                if new_o - o > 0:
                    trans, o = new_trans, new_o
                else:
                    p = math.exp((new_o - o) / T)
                    r = np.random.uniform(low=0, high=1)
                    if r < p:
                        trans, o = new_trans, new_o

                print(T, i, trans, o)

        T *= 0.6  # 降温函数，也可使用T=0.9T
    return trans


def get_mocap_to_lidar_translation(mocap_points, lidar_points, lidar_to_mocap_rotation):
    source_points = mocap_points.astype(np.float32)
    target_points = affine(
        lidar_points, lidar_to_mocap_rotation).astype(np.float32)
    target_kdtree = pcl.PointCloud(target_points).make_kdtree_flann()
    return icp_without_rotation(source_points, target_points, target_kdtree)

def mqh_get_mocap_to_lidar_translation(mocap_points, lidar_points, lidar_to_mocap_rotation):
    source_points = mocap_points.astype(np.float32)
    target_points = affine(
        lidar_points, lidar_to_mocap_rotation).astype(np.float32)
    source_kdtree = pcl.PointCloud(source_points).make_kdtree_flann()
    return -icp_without_rotation(target_points, source_points, source_kdtree)


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
