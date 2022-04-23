from collections import namedtuple

from io3d import mocap, pcd, ply
from io3d.pcd import read_point_cloud, save_point_cloud
from scipy.spatial.transform import Rotation as R
from smpl import model as smpl_model
from tqdm import tqdm
from typing import Dict, List
from util import mocap_util, path_util, pc_util, img_util, transformation, multiprocess

import argparse
import functools
import json
import logging
import numpy as np
import os
import pandas as pd
import shutil
import smpl.generate_ply

IMAGE_FRAME_RATE = 29.83
POINTCLOUD_FRAME_RATE = 10
MOCAP_FRAME_RATE = 100
MAX_PROCESS_COUNT = 48
MIN_PERSON_POINTS_NUM = 20

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s', datefmt='%b %d %H:%M:%S')
logger = logging.getLogger()


def dict_to_struct(d):
    return namedtuple('Struct', d.keys())(*d.values())


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


def prepare_current_dirs(raw_dir, dataset_dirs, index):
    cur_dirs = {'raw_dir': os.path.join(raw_dir, str(index))}
    for key, value in dataset_dirs.items():
        if key == 'calib_dir':
            cur_dirs[key] = value
        else:
            cur_dirs[key] = os.path.join(value, str(index))
        os.makedirs(cur_dirs[key], exist_ok=True)
    return dict_to_struct(cur_dirs)


def generate_keypoints(images_dir, keypoints_dir, openpose_path):
    # openpose
    origin_cwd = os.getcwd()
    openpose_bin_path = os.path.join(
        openpose_path, 'build/examples/openpose/openpose.bin')
    cmd = '{} --image_dir {} --write_json {} --display 0 --render_pose 0'.format(
        openpose_bin_path, images_dir, keypoints_dir)
    os.chdir(openpose_path)
    os.system(cmd)
    os.chdir(origin_cwd)


def generate_depth_images(segment_dir, depth_dir):
    path_util.clear_folder(depth_dir)
    bin_path = os.path.join('/home/ljl/ASC_Lidar_human/bin/make_range_image')
    cmd = '{} --in_dir {} --out_dir {} --resolution 0.2'.format(
        bin_path, segment_dir, depth_dir)
    print(cmd)
    os.system(cmd)

# 因为主函数需要并行，所以子函数必须串行


def generate_segment(pc_dir: str,
                     pc_indexes: np.ndarray,
                     segment_dir: str,
                     bg_points: np.ndarray,
                     crop_box: np.ndarray):
    path_util.clear_folder(segment_dir)
    pc_filenames = path_util.get_sorted_filenames_by_index(pc_dir)
    pre_center = None
    reserved = []
    print(pc_dir)
    for pc_filename in pc_filenames:
        if path_util.get_index(pc_filename) not in pc_indexes:
            continue
        bg_kdtree = pc_util.get_kdtree(
            pc_util.crop_points(bg_points, crop_box))
        lidar_points = pcd.read_point_cloud(pc_filename)[:, :3]
        lidar_points = pc_util.crop_points(lidar_points, crop_box)
        lidar_points = pc_util.erase_background(
            lidar_points, bg_kdtree, pre_center)
        pre_center = np.mean(lidar_points, axis=0)
        save_point_cloud(os.path.join(segment_dir, pc_filename), lidar_points)

        cur_reserved = lidar_points.shape[0] >= MIN_PERSON_POINTS_NUM
        reserved.append(cur_reserved)
        if cur_reserved:
            # basename = os.path.basename(pc_filename)
            # basename = os.path.splitext(basename)[0] + '.ply'
            # ply.save_point_cloud(os.path.join(
            #     segment_dir, basename), lidar_points)
            pre_center = np.mean(lidar_points, axis=0)
    logger.info('Generate segmented point clouds finished')
    return reserved


def save_smpl_data(mocap_points, cur_translation, pose, seg_filename, beta, lidar_to_mocap_RT, pose_dir, mesh_gen):
    mocap_to_lidar_RT = np.linalg.inv(lidar_to_mocap_RT)
    mocap_to_lidar_R = mocap_to_lidar_RT[:3, :3]
    # mocap_to_lidar_T = mocap_to_lidar_RT[:3, 3].reshape(3, )
    # origin_mocap_points = mocap_points.copy()
    mocap_points = transformation.mocap_to_lidar(
        mocap_points, lidar_to_mocap_RT, translation=cur_translation)
    index_str = os.path.splitext(os.path.basename(seg_filename))[0]
    smpl.generate_ply.save_ply(mocap_points, os.path.join(
        pose_dir, '{}.ply'.format(index_str)))
    pose[0:3] = (R.from_matrix(mocap_to_lidar_R)
                 * R.from_rotvec(pose[0:3])).as_rotvec()
    # trans = R.from_matrix(mocap_to_lidar_R).apply(
    #     origin_to_root_translation + cur_translation) + mocap_to_lidar_T
    # 因为smpl的全局旋转的中心不是原点，所以旋转完会有一定的偏移量，这边做个补偿
    trans = mocap_points[0] - mesh_gen.get_vertices(pose, beta)[0]
    # trans = mocap_points[0] - origin_mocap_points[0]
    # trans += mocap_points[0] - \
    #     smpl_model.get_vertices(pose, beta, trans)[0]
    with open(os.path.join(pose_dir, '{}.json'.format(index_str)), 'w') as f:
        d = {'beta': beta.tolist(),
             'pose': pose.tolist(),
             'trans': trans.tolist()}
        f.write(json.dumps(d))


def save_smpl(filename, vertices, joints):
    np.savez(filename, vertices=vertices, joints=joints)


def read_smpl(filename):
    data = np.load(filename)
    return data['vertices'], data['joints']


def generate_smpl(cur_dirs: Dict[str, str],
                  mocap_data: mocap.MoCapData,
                  mocap_indexes: List[int],
                  cur_process_info):
    start_index = cur_process_info['start_index']
    end_index = cur_process_info['end_index']
    pc_start_index = start_index['pointcloud']
    mocap_start_index = start_index['mocap']
    total_lenth = end_index-pc_start_index+1
    mocap_indexes = mocap_indexes[:total_lenth]

    poses = [mocap_data.pose(mocap_index) for mocap_index in mocap_indexes]
    print(len(poses))
    # indexes = [path_util.get_index(
    #     filename) for filename in path_util.get_sorted_filenames_by_index(cur_dirs.segment_dir, False)]
    indexes = range(pc_start_index, end_index+1)
    print(len(indexes))
    n_poses = len(poses)
    assert n_poses == len(indexes)
    batch_size = 512
    n_batch = (n_poses + batch_size - 1) // batch_size
    smpl = smpl_model.SMPL()
    vertices = np.zeros((0, 6890, 3))
    joints = np.zeros((0, 24, 3))
    for i in range(n_batch):
        lb = i * batch_size
        ub = min((i + 1) * batch_size, n_poses)
        cur_poses = np.stack(poses[lb:ub])
        cur_vertices = smpl.get_vertices(pose=cur_poses)
        cur_joints = smpl.get_joints(cur_vertices)
        vertices = np.concatenate((vertices, cur_vertices))
        joints = np.concatenate((joints, cur_joints))
        print('{}/{}'.format(i, n_batch))
    filenames = [os.path.join(
        cur_dirs.smpl_dir, '{}.npz'.format(index)) for index in indexes]
    path_util.clear_folder(cur_dirs.smpl_dir)
    for a, b, c in tqdm(zip(filenames, vertices, joints), total=len(filenames)):
        save_smpl(a, b, c)

    # multiprocess.multi_func(save_smpl, MAX_PROCESS_COUNT, len(
    #     filenames), 'saving smpl vertices and joints', False, filenames, vertices, joints)


def generate_pose(cur_dirs: Dict[str, str],
                  mocap_data: mocap.MoCapData,
                  mocap_indexes: List[int],
                  cur_process_info: Dict):
    start_index = cur_process_info['start_index']
    end_index = cur_process_info['end_index']
    pc_start_index = start_index['pointcloud']
    mocap_start_index = start_index['mocap']
    total_lenth = end_index-pc_start_index+1
    mocap_indexes = mocap_indexes[:total_lenth]

    segment_filenames = path_util.get_sorted_filenames_by_index(
        cur_dirs.segment_dir)
    segment_filenames=segment_filenames[pc_start_index-1:end_index]
    # Python中list的append操作是线程安全的，所以可以append的时候带上索引，然后按照索引排序
    n = len(segment_filenames)

    lidar_to_mocap_RT = np.array(
        cur_process_info['lidar_to_mocap_RT']).reshape(4, 4)

    if 'beta' in cur_process_info:
        beta = np.array(cur_process_info['beta'])
    else:
        beta = np.zeros((10, ))

    logger.info('Calculate MoCap to LiDAR translations')

    segment_point_clouds = multiprocess.multi_func(
        pcd.read_point_cloud, MAX_PROCESS_COUNT, n, 'read segment points', True, segment_filenames)

    smpl_filenames = path_util.get_sorted_filenames_by_index(cur_dirs.smpl_dir)
    mocap_point_clouds, mocap_joints = multiprocess.multi_func(
        read_smpl, MAX_PROCESS_COUNT, n, 'read smpl vertices and joints', True, smpl_filenames)

    poses = []
    mocap_to_lidar_translations = []

    poses = [mocap_data.pose(mocap_index)
             for mocap_index in tqdm(mocap_indexes)]
    mocap_to_lidar_translations = [transformation.get_mocap_to_lidar_translation(
        mocap_points, segment_points[:, :3], lidar_to_mocap_RT) for mocap_points, segment_points in tqdm(zip(mocap_point_clouds, segment_point_clouds), total=n)]

    # 平滑平移量
    half_width = 10
    translation_sum = np.zeros((3, ))
    l = 0
    r = 0
    cnt = 0
    aux = []
    for i in range(n):
        rb = min(n - 1, i + half_width)
        lb = max(0, i - half_width)
        while r <= rb:
            translation_sum += mocap_to_lidar_translations[r]
            cnt += 1
            r += 1
        while l < lb:
            translation_sum -= mocap_to_lidar_translations[l]
            cnt -= 1
            l += 1
        aux.append(translation_sum / cnt)
    mocap_to_lidar_translations = aux

    # mocap_to_lidar_translations是把点云场景放平之后mocap数据到点云数据的平移
    # 因此都可以以第一帧的T_z为准
    for i in range(n):
        mocap_to_lidar_translations[i][2] = mocap_to_lidar_translations[0][2]

    path_util.clear_folder(cur_dirs.pose_dir)
    smpl = smpl_model.SMPL()

    for mocap_points, cur_translation, pose, seg_filename in tqdm(zip(mocap_point_clouds, mocap_to_lidar_translations, poses, segment_filenames), total=len(poses)):
        save_smpl_data(mocap_points, cur_translation, pose, seg_filename,
                       beta, lidar_to_mocap_RT, cur_dirs.pose_dir, smpl)

    # multiprocess.multi_func(
    #     functools.partial(save_smpl_data, beta=beta,
    #                       lidar_to_mocap_RT=lidar_to_mocap_RT, pose_dir=, mesh_gen=smpl),
    #     MAX_PROCESS_COUNT, len(mocap_point_clouds), "Saving smpl data", False, mocap_point_clouds, mocap_to_lidar_translations, poses, segment_filenames)


def read_array_dat(filename):
    import struct
    array = []
    with open(filename, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        for i in range(n):
            array.append(struct.unpack('d', f.read(8))[0])
    return array


def process_each(index, args):
    # os.sched_setaffinity(os.getpid(), [cpu_id])
    # parse args
    raw_dir = args.raw_dir
    dataset_dir = args.dataset_dir

    openpose_path = args.openpose_path

    # parse generate params
    gen = {}
    for k, v in vars(args).items():
        if k.startswith('gen') and k != 'gen_all':
            gen[k[4:]] = v or args.gen_all
    gen = dict_to_struct(gen)

    dataset_dirs = prepare_dataset_dirs(dataset_dir)

    # read json
    with open(os.path.join(raw_dir, 'process_info.json')) as f:
        process_info = json.load(f)

    cur_dirs = prepare_current_dirs(raw_dir, dataset_dirs, index)
    logger.info('index: {}'.format(index))

    video_path = path_util.get_one_path_by_suffix(cur_dirs.raw_dir, '.mp4')
    bvh_path = path_util.get_one_path_by_suffix(cur_dirs.raw_dir, '.bvh')
    # 每个文件夹下面有两个pcap，一个是背景的pcap，另一个是实际数据的pcap
    pcap_paths = path_util.get_paths_by_suffix(cur_dirs.raw_dir, '.pcap')
    if len(pcap_paths) == 1:
        pcap_path = pcap_paths[0]
    elif len(pcap_paths) == 2:
        pcap_path = pcap_paths[0] if 'bg.pcap' not in pcap_paths[0] else pcap_paths[1]
    else:
        logger.disabled = False
        logger.error('the number of pcaps is wrong')
        exit(1)
    img_dir = cur_dirs.images_dir
    pc_dir = cur_dirs.pointclouds_dir
    mocap_dir = cur_dirs.mocaps_dir
    segment_dir = cur_dirs.segment_dir
    cur_process_info = process_info[str(index)]

    mocap_indexes_path = os.path.join(mocap_dir, 'mocap_indexes.npy')
    image_indexes_path = os.path.join(img_dir, 'image_indexes.npy')

    if gen.basic:
        # 产生背景点云
        #bg_points_path = os.path.join(cur_dirs.raw_dir, 'bg.pcd')
        #bg_points = pcd.read_point_cloud(bg_points_path)

        # img_util.video_to_images(video_path, img_dir)
        # # path_util.clear_folder(pc_dir)
        #pc_util.pcap_to_pcds(pcap_path, pc_dir)
        mocap_util.get_csvs_from_bvh(bvh_path, mocap_dir)

        pc_timestamps = np.array(read_array_dat(
            os.path.join(pc_dir, 'timestamps.dat')))


        # # 抽帧对齐
        # # img_frame_nums = len(os.listdir(img_dir))
        # pc_frame_nums = len(os.listdir(pc_dir)) - 1  # 需要排除时间戳的文件
        # mocap_frame_nums = pd.read_csv(
        #     path_util.get_one_path_by_suffix(mocap_dir, '_worldpos.csv')).shape[0]

        # 得到最多重叠的帧数
        if 'key_indexes' in cur_process_info:
            key_indexes = cur_process_info['key_indexes']
            pc_start_index = key_indexes[0][0]
            pc_end_index = key_indexes[-1][0]

            pc_indexes = np.arange(pc_start_index, pc_end_index + 1)
            mocap_indexes = []
            for i in range(len(key_indexes) - 1):
                p1, m1 = key_indexes[i]
                p2, m2 = key_indexes[i + 1]
                if i + 2 == len(key_indexes):
                    p2 += 1
                cur_timestamps = pc_timestamps[p1 - 1:p2 - 1]
                cur_timestamps = cur_timestamps - cur_timestamps[0]
                duration = pc_timestamps[p2 - 1] - pc_timestamps[p1 - 1]
                mocap_fps = (m2 - m1) / duration
                mocap_indexes += (cur_timestamps * mocap_fps +
                                  m1).astype(int).tolist()

        else:
            pc_start_index = cur_process_info['start_index']['pointcloud']
            mocap_start_index = cur_process_info['start_index']['mocap']
            pc_end_index = cur_process_info['end_index']
            pc_frame_nums = len(os.listdir(pc_dir)) - 1
            mocap_frame_nums = pd.read_csv(
                path_util.get_one_path_by_suffix(mocap_dir, '_worldpos.csv')).shape[0]
            if pc_end_index == -1:
                pc_end_index = pc_frame_nums + 1
            pc_indexes = np.arange(pc_start_index, pc_end_index)
            pc_timestamps = pc_timestamps[pc_indexes - 1]
            pc_timestamps -= pc_timestamps[0]
            mocap_indexes = MOCAP_FRAME_RATE * pc_timestamps + mocap_start_index
            mocap_indexes = mocap_indexes[mocap_indexes < mocap_frame_nums]
            n_frames = min(pc_indexes.shape[0], mocap_indexes.shape[0])
            pc_indexes = pc_indexes[:n_frames]

        # img_indexes = np.arange(
        #     img_start_index, img_frame_nums + 1, IMAGE_FRAME_RATE // POINTCLOUD_FRAME_RATE)
        # mocap_indexes = np.arange(
        #     mocap_start_index - 1, mocap_frame_nums, MOCAP_FRAME_RATE // POINTCLOUD_FRAME_RATE)
        # print(mocap_indexes)
        # n_frames = min(pc_indexes.shape[0], min(
        #     img_indexes.shape[0], mocap_indexes.shape[0]))
        # motionbuilder中的索引是从0开始，dataframe也是从0开始，两者的第0帧都是T pose
        # img_indexes = img_indexes[:n_frames]
        # path_util.remove_unnecessary_frames(img_dir, img_indexes)
        # path_util.remove_unnecessary_frames(pc_dir, pc_indexes)

        # 如果人不在划定区域内则不考虑，将相应的帧去掉
        # reserved = generate_segment(
        #     pc_dir, pc_indexes, segment_dir, bg_points, cur_process_info['box'])
        # pc_indexes = pc_indexes[reserved]
        # img_indexes = img_indexes[reserved]
        # path_util.remove_unnecessary_frames(img_dir, img_indexes)
        # path_util.remove_unnecessary_frames(pc_dir, pc_indexes)

        # pc_indexes = [path_util.get_index(
        #     filename) for filename in path_util.get_sorted_filenames_by_index(cur_dirs.segment_dir, False)]
        # key_indexes = cur_process_info['key_indexes']
        # pc_indexes = [pc_index in range(
        #     key_indexes[0][0], key_indexes[-1][0] + 1) for pc_index in pc_indexes]
        # path_util.remove_unnecessary_frames(cur_dirs.segment_dir, pc_indexes)

        # # 按照pedx数据集，将点云全部用ply存储
        # for pcd_filename in path_util.get_sorted_filenames_by_index(pc_dir):
        #     ply_filename = pcd_filename.replace('pcd', 'ply')
        #     ply.pcd_to_ply(pcd_filename, ply_filename)
        #     os.remove(pcd_filename)

        return

    pc_indexes = np.array([path_util.get_index(
        filename) for filename in path_util.get_sorted_filenames_by_index(segment_dir, False)])

    if gen.image_indexes:

        pc_timestamps = np.array(read_array_dat(
            os.path.join(pc_dir, 'timestamps.dat')))

        image_start_index = cur_process_info['start_index']['image']
        pc_timestamps = pc_timestamps[pc_indexes - 1]
        pc_timestamps -= pc_timestamps[0]
        image_indexes = IMAGE_FRAME_RATE * pc_timestamps + image_start_index
        img_dir = '/cwang/home/ljl/data/lidarcap/images/{}'.format(index)
        image_frame_nums = len(
            list(filter(lambda x: x.endswith('.png'), os.listdir(img_dir))))
        image_frame_nums = max(image_frame_nums, len(
            list(filter(lambda x: x.endswith('.jpg'), os.listdir(img_dir)))))
        print(image_frame_nums)
        print(image_indexes)
        image_indexes = np.around(
            image_indexes[image_indexes < image_frame_nums]).astype(int)
        print(image_indexes)
        np.save(image_indexes_path, image_indexes)

    if gen.mocap_indexes:
        pc_timestamps = np.array(read_array_dat(
            os.path.join(pc_dir, 'timestamps.dat')))
        if 'key_indexes' in cur_process_info:
            mocap_indexes = []
            key_indexes = cur_process_info['key_indexes']
            for i in range(len(key_indexes) - 1):
                p1, m1 = key_indexes[i]
                p2, m2 = key_indexes[i + 1]
                if i + 2 == len(key_indexes):
                    p2 += 1
                cur_timestamps = pc_timestamps[p1 - 1:p2 - 1]
                cur_timestamps = cur_timestamps - cur_timestamps[0]
                duration = pc_timestamps[p2 - 1] - pc_timestamps[p1 - 1]
                mocap_fps = (m2 - m1) / duration
                mocap_indexes += np.around((cur_timestamps * mocap_fps +
                                            m1)).astype(int).tolist()
            mocap_indexes = np.array(mocap_indexes)
        else:
            start_index = cur_process_info['start_index']
            end_index = cur_process_info['end_index']
            pc_start_index = start_index['pointcloud']
            total_lenth = end_index - pc_start_index + 1

            mocap_start_index = cur_process_info['start_index']['mocap']
            mocap_frame_nums = pd.read_csv(
                path_util.get_one_path_by_suffix(mocap_dir, '_worldpos.csv')).shape[0]
            pc_timestamps = pc_timestamps[pc_indexes - 1]
            pc_timestamps -= pc_timestamps[0]
            mocap_indexes = MOCAP_FRAME_RATE * pc_timestamps + mocap_start_index
            mocap_indexes = np.around(
                mocap_indexes[mocap_indexes < mocap_frame_nums]).astype(int)
            mocap_indexes = mocap_indexes[:total_lenth]
        print(len(mocap_indexes))
        np.save(mocap_indexes_path, mocap_indexes)

    if not os.path.exists(mocap_indexes_path):
        logger.info('NO MOCAP INDEXES')
        return

    mocap_indexes = np.load(mocap_indexes_path)

    if gen.segment:
        bg_points_path = os.path.join(cur_dirs.raw_dir, 'bg.pcd')
        bg_points = pcd.read_point_cloud(bg_points_path)
        generate_segment(pc_dir, segment_dir, bg_points,
                         cur_process_info['box'])

    if gen.keypoints:
        generate_keypoints(img_dir, cur_dirs.keypoints_dir, openpose_path)

    if gen.mask:
        import mask_rcnn.inference
        mask_rcnn.inference.inference(
            img_dir, cur_dirs.bbox_dir, cur_dirs.mask_dir)

    if gen.smpl:
        worldpos_csv = path_util.get_one_path_by_suffix(
            mocap_dir, '_worldpos.csv')
        rotation_csv = path_util.get_one_path_by_suffix(
            mocap_dir, '_rotations.csv')
        mocap_data = mocap.MoCapData(worldpos_csv, rotation_csv)
        generate_smpl(cur_dirs, mocap_data, mocap_indexes,cur_process_info)

    if gen.pose:
        worldpos_csv = path_util.get_one_path_by_suffix(
            mocap_dir, '_worldpos.csv')
        rotation_csv = path_util.get_one_path_by_suffix(
            mocap_dir, '_rotations.csv')
        mocap_data = mocap.MoCapData(worldpos_csv, rotation_csv)
        generate_pose(cur_dirs, mocap_data, mocap_indexes, cur_process_info)

    if gen.depth_images:
        generate_depth_images(segment_dir, cur_dirs.depth_dir)

    # project(cur_dirs, mocap.MoCapData(worldpos_csv, rotation_csv),
    #         bg_points, cur_process_info['box'], mocap_indexes)


def main(args):

    import re
    if re.match('^([1-9]\d*)-([1-9]\d*)$', args.index):
        start_index, end_index = args.index.split('-')
        indexes = list(range(int(start_index), int(end_index) + 1))
    elif re.match('^(([1-9]\d*),)*([1-9]\d*)$', args.index):
        indexes = [int(x) for x in args.index.split(',')]
    else:
        indexes = [int(args.index)]

    if args.gen_pose or len(indexes) == 1:

        for index in indexes:
            process_each(index, args)
    else:
        multiprocess.multi_func(
            functools.partial(process_each, args=args),
            4, len(indexes), 'process each', True, indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True)

    parser.add_argument('--raw_dir', type=str,
                        default='/SAMSUMG8T/ljl/zjy/raw') # /cwang/home/ljl/data/raw
    parser.add_argument('--dataset_dir', type=str,
                        default='/SAMSUMG8T/ljl/zjy/lidarcap') # /cwang/home/ljl/data/lidarcap
    parser.add_argument('--openpose_path', type=str,
                        default='/home/ljl/Tools/openpose')

    parser.add_argument('--log', action='store_true')

    parser.add_argument('--gen_all', action='store_true',
                        help='generate the images, point cloud files and csv files')
    parser.add_argument('--gen_basic', action='store_true')
    parser.add_argument('--gen_keypoints', action='store_true')
    parser.add_argument('--gen_mask', action='store_true')
    parser.add_argument('--gen_pose', action='store_true')
    parser.add_argument('--gen_segment', action='store_true')
    parser.add_argument('--gen_mocap_indexes', action='store_true')
    parser.add_argument('--gen_smpl', action='store_true')
    parser.add_argument('--gen_image_indexes', action='store_true')
    parser.add_argument('--gen_depth_images', action='store_true')
    # parser.add_argument('--write_smpl_vertices', action='store_true')

    args = parser.parse_args()
    if not args.log:
        logger.disabled = True
    main(args)
