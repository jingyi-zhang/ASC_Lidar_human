from collections import Counter
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import numpy as np
import os
import pcl
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util import path_util

from io3d.pcd import read_point_cloud, save_point_cloud

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))


def crop_points(points, crop_box):
    x_min, y_min = crop_box['min']
    x_max, y_max = crop_box['max']
    mask = np.logical_and(points[:, 0] > x_min, points[:, 1] > y_min)
    mask = np.logical_and(mask, points[:, 0] < x_max)
    mask = np.logical_and(mask, points[:, 1] < y_max)
    return points[mask].copy()


def pcap_to_pcds(pcap_path, pcds_dir):
    assert os.path.isabs(pcap_path) and os.path.isfile(pcap_path)
    assert os.path.isabs(pcds_dir) and os.path.isdir(pcds_dir)
    read_pcap_bin_path = os.path.join(ROOT_PATH, 'bin', 'read_pcap')
    os.system(
        '{} --in_file {} --out_dir {}'.format(read_pcap_bin_path, pcap_path, pcds_dir))


# 聚类
def dbscan_outlier_removal(points, pre_center):
    labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(points)
    sort_labels = [x[0] for x in Counter(labels).most_common()]
    if -1 in sort_labels:
        sort_labels.remove(-1)
    for label in sort_labels:
        cur_points = points[labels == label]
        # 初始的聚类，人是站直的，所以可以根据z的高度来把不是人的排除
        if pre_center is None:
            z_min = min(cur_points[:, 2])
            z_max = max(cur_points[:, 2])
            if z_max - z_min < 1.4:
                print('too low!')
                exit()
                continue
            return points[labels == label]
        # 保证人是连续移动的，相邻两帧的位置相差不超过一定阈值
        else:
            cur_center = np.mean(cur_points, axis=0)
            if np.linalg.norm(cur_center - pre_center) > 0.5:
                continue
            return points[labels == label]
    return points[0].reshape(1, -1)


def erase_background(points, bg_kdtree, pre_center):
    EPSILON = 0.12
    EPSILON2 = EPSILON ** 2
    squared_distance = bg_kdtree.nearest_k_search_for_cloud(
        pcl.PointCloud(points), 1)[1].flatten()
    erased_points = points[squared_distance > EPSILON2]
    if erased_points.shape[0] == 0:
        erased_points = points[0].reshape(1, -1)
    return erased_points
    # return dbscan_outlier_removal(erased_points, pre_center)


def get_kdtree(points):
    return pcl.PointCloud(points.astype(np.float32)).make_kdtree_flann()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcap2pcd', action='store_true', default=False)
    parser.add_argument('--remove_bg', action='store_true', default=False)
    parser.add_argument('--pcap_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--bg_path', type=str, default=None)
    args = parser.parse_args()

    if args.pcap2pcd:
        pcap_path = args.pcap_path
        save_path = args.save_path
        pcap_to_pcds(pcap_path, save_path)

    if args.remove_bg:
        bg_path = '/SAMSUMG8T/ljl/zjy/raw/417/bg.pcd'
        bg_points = read_point_cloud(bg_path)
        kdtree = get_kdtree(bg_points)

        pc_path = '/SAMSUMG8T/ljl/zjy/lidarcap/pointclouds/417'
        human_path = '/SAMSUMG8T/ljl/zjy/lidarcap/labels/3d/segment/417'
        os.makedirs(human_path, exist_ok=True)
        pre_center = None


        filenames = path_util.get_sorted_filenames_by_index(pc_path, isabs=False)

        for filename in tqdm(filenames):
            if (int(os.path.splitext(filename)[0])) < 100:
                continue

            file_path = os.path.join(pc_path, filename)
            save_path = os.path.join(human_path, filename)
            points = read_point_cloud(file_path)

            human_points = erase_background(points, kdtree, pre_center)
            pre_center = np.mean(human_points, axis=0)
            save_point_cloud(save_path, human_points)
