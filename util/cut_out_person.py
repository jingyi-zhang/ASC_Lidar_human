import os

import numpy as np


mqh_work_path = '/SAMSUMG8T/mqh/417'


"""
pcap_path = '/SAMSUMG8T/sxl/Project/pcap'
import glob, os
from util.pc_util import pcap_to_pcds
for f in glob.glob(f'{pcap_path}/*.pcap'):
    if 'wx' in f or 'dyd' in f:
        pcds_path = os.path.join(mqh_work_path, 'pcap_pcds', os.path.basename(f).split('.')[0])
        os.makedirs(pcds_path, exist_ok=True)
        pcap_to_pcds(f, pcds_path)

"""


def cut_out_person(input_pcds_path, output_pcds_path):
    import open3d as o3d
    import tqdm
    import glob, os

    assert os.path.exists(input_pcds_path), 'input_pcds_path不存在！'
    print(f'cut out person from {input_pcds_path} to {output_pcds_path}')

    filenames = glob.glob(os.path.join(input_pcds_path, '*.pcd'))


    #/SAMSUMG8T/sxl/Project/pcap
    #"""
    filenames.sort(key=lambda e: int(os.path.basename(e).split('.')[0]))


    outliers = []
    for f in tqdm.tqdm(filenames, desc='Remove Ground', ncols=60):
        pcd = o3d.io.read_point_cloud(f)
        p = np.asarray(pcd.points)
        p = p[np.logical_and(np.logical_and(-4.2 < p[:, 1], p[:, 1] < 2.3), p[:, 0] > 1)]
        pcd.points = o3d.utility.Vector3dVector(p)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        outliers.append(np.asarray(outlier_cloud.points))

    persons = []
    center_pos = outliers[0].mean(axis=0, keepdims=True)
    for p in tqdm.tqdm(outliers, desc='Cut Out Person', ncols=60):
        p = p[((p - center_pos) ** 2).sum(axis=1) < 2 ** 2]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=False))
        label_with_max_points = np.bincount(labels + 1).argmax() - 1
        assert label_with_max_points != -1, '在聚类后具有最多点的标签是noice!考虑提高cluster_dbscan的eps值'
        p = p[labels==label_with_max_points]
        center_pos = p.mean(axis=0, keepdims=True)
        persons.append(p)

    """
    #可视化
    from PclVisual import PV
    i = 0
    i += 10
    pv.remove_all_clouds()
    
    pv.add_cloud(np.asarray(o3d.io.read_point_cloud(filenames[i]).points).tolist(), [[255, 255, 255]], 'points')
    pv.add_cloud(outliers[i].tolist(), [[0, 0, 255]], 'outliners')
    pv.add_cloud(persons[i].tolist(), [[0, 255, 0]], 'person').send()
    """

    for p, f in zip(persons, filenames):
        pcd_file_name = os.path.join(output_pcds_path, f"{os.path.basename(f).split('.')[0]}.pcd")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        o3d.io.write_point_cloud(pcd_file_name, pcd)
    print('Finish!')


"""
for dir in os.listdir(mqh_work_path):
    pcap_pcds_path = os.path.join(mqh_work_path, dir, 'pcap_pcds')
    person_pcds_path = os.path.join(mqh_work_path, dir, 'person_pcds')
    os.makedirs(person_pcds_path, exist_ok=True)
    cut_out_person(pcap_pcds_path, person_pcds_path)
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcap_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    cut_out_person(args.pcap_path, args.save_path)

    #--pcap_path /SAMSUMG8T/mqh/417/4_17_14_57_wx_jiandongxi/pcap_pcds --save_path /SAMSUMG8T/mqh/417/4_17_14_57_wx_jiandongxi/person_pcds
