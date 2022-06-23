import asyncio
import pickle

import h5py
import open3d as o3d

import numpy as np

import sys

import socketio
import tornado
sys.path.append('/cwang/home/ljl/anaconda3/envs/torch/lib/python3.7/site-packages/site-packages/python_pcl-0.3.0rc1-py3.7-linux-x86_64.egg')
# 导入ljl的pcl环境

import os
from functools import reduce
import tqdm
import json

from io3d import mocap, pcd
from seq_process_pipeline_server.pipeline_util import read_array_dat, affine
from util import path_util, mocap_util, multiprocess, transformation
from seq_process_pipeline_server import pipeline_util

from seq_process_pipeline_server.pipeline import seq_process
from seq_process_pipeline_server.pipeline_util import save_smpl_json, fix_points_num
from smpl.generate_ply import save_ply
from smpl import model as smpl_model

MAX_PROCESS_COUNT = 32

import torch
torch.set_num_threads(1)
#   定义一个序列类，其中包含数据处理的各种算法（切分人体，配准等），输入是我们采集的动捕序列和点云，输出是数据集

class server:
    def __init__(self):
        self.seq = seq_process(None, None)
        self.seq_cache = {}
        self.smpl = smpl_model.SMPL()

    def init(self, raw_dir, dataset_dir, index):
        dataset_dirs = pipeline_util.prepare_dataset_dirs(dataset_dir)

        cur_dirs = pipeline_util.prepare_current_dirs(raw_dir, dataset_dirs, index)
        print('index: {}'.format(index))

        #video_path = path_util.get_one_path_by_suffix(cur_dirs.raw_dir, '.mp4')

        self.cur_dirs = cur_dirs
        self.index = str(index)

        if self.index in self.seq_cache:
            self.seq = self.seq_cache[self.index]
        elif os.path.exists(os.path.join(s.cur_dirs.raw_dir, 'pipeline.npy')):
            with open(os.path.join(s.cur_dirs.raw_dir, 'pipeline.npy'), 'rb') as f:
                print(f'load seq_process from disk')
                self.seq = pickle.load(f)
                self.seq.pcds = self.get_raw_pc()
                self.seq.mocaps = self.get_raw_mocap()
                if not hasattr(self.seq, 'person_label_list'):
                    self.seq.person_label_list = [None for i in range(len(self.seq.pcds))]
                if not hasattr(self.seq, 'object_rt_list'):
                    self.seq.object_rt_list = [None for i in range(len(self.seq.pcds))]
                if not hasattr(self.seq, 'obj'):
                    self.seq.obj = None
                if not hasattr(self.seq, 'mocap_pose_cache'):
                    self.seq.mocap_pose_cache = {}
        else:
            self.seq = seq_process(self.get_raw_pc(), self.get_raw_mocap())
            self.seq_cache[self.index] = self.seq

        #self.test_mocap_timestamp()

    def get_process_info(self):
        pc_start_index = self.seq.to_raw_pcd_index(0)
        pc_end_index = self.seq.to_raw_pcd_index(self.seq.get_pcds_len() - 1)
        mocap_start_index = self.seq.to_raw_mocap_index(0)
        lidar_to_mocap_RT = self.seq.get_cur_transform().tolist()

        cur_process_info = {'start_index': {'image': 0, 'pointcloud': pc_start_index, 'mocap': mocap_start_index},
                            'box': {'min': [0, 0], 'max': [0, 0]}, 'lidar_to_mocap_RT': lidar_to_mocap_RT,
                            'end_index': pc_end_index}
        return cur_process_info

    def load_process_info(self):
        with open(os.path.join(os.path.dirname(self.cur_dirs.raw_dir), 'process_info.json')) as f:
            process_info = json.load(f)
        if str(self.index) in process_info:
            info = process_info[str(self.index)]
            print('load existed info:', info)
            self.seq.apply_pcds_map(np.arange(info['start_index']['pointcloud'], info['end_index'], 1))
            self.seq.apply_mocaps_map(np.arange(info['start_index']['mocap'], self.seq.get_mocaps_len(), 1))
            #self.gen_pc_and_mocap_indexes()
            self.seq.apply_transform(np.array(info['lidar_to_mocap_RT']).reshape(4, 4))
        else:
            print('no existed info to load.')

    def apply_transform_from_process_info(self, index):
        with open(os.path.join(os.path.dirname(self.cur_dirs.raw_dir), 'process_info.json')) as f:
            process_info = json.load(f)
        if str(index) in process_info:
            info = process_info[str(index)]
            self.seq.apply_transform(np.array(info['lidar_to_mocap_RT']).reshape(4, 4))
        else:
            print(f'no existed info for index:{index} to load.')

    def write_process_info(self):
        print("DEBUG模式，暂不保存process_info!")
        return
        with open(os.path.join(os.path.dirname(self.cur_dirs.raw_dir), 'process_info.json')) as f:
            process_info = json.load(f)

        with open(os.path.join(os.path.dirname(self.cur_dirs.raw_dir), 'process_info.json'), 'w') as f:
            process_info[str(self.index)] = self.get_process_info()
            json.dump(process_info, f)

    def write_seq_process(self):
        print("DEBUG模式，暂时不保存pipeline.npy!")
        return
        with open(os.path.join(s.cur_dirs.raw_dir, 'pipeline.npy'), 'wb') as f:
            pickle.dump(s.seq, f)
        print('success save seq pipeline!')

    def get_raw_pc(self):
        filenames = path_util.get_sorted_filenames_by_index(self.cur_dirs.pointclouds_dir)
        if len(filenames) == 0:
            pcap_paths = path_util.get_paths_by_suffix(self.cur_dirs.raw_dir, '.pcap')
            save_path = self.cur_dirs.pointclouds_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            from util.pc_util import pcap_to_pcds
            print(f'pcap2pcd from {pcap_paths[0]} to {save_path}')
            pcap_to_pcds(pcap_paths[0], save_path)
            filenames = path_util.get_sorted_filenames_by_index(self.cur_dirs.pointclouds_dir)

        point_clouds = multiprocess.multi_func(
            pcd.read_point_cloud, 32, len(filenames), 'read raw point clouds:', True, filenames)
        return point_clouds

    def get_raw_mocap(self):
        self.gen_csv()
        worldpos_csv = path_util.get_one_path_by_suffix(
            self.cur_dirs.mocaps_dir, '_worldpos.csv')
        rotation_csv = path_util.get_one_path_by_suffix(
            self.cur_dirs.mocaps_dir, '_rotations.csv')
        mocap_data = mocap.MoCapData(worldpos_csv, rotation_csv)
        print(f'load mocap_data:({len(mocap_data)})')
        return mocap_data

    def gen_csv(self):
        bvh_path = path_util.get_one_path_by_suffix(self.cur_dirs.raw_dir, '.bvh')
        mocap_util.get_csvs_from_bvh(bvh_path, self.cur_dirs.mocaps_dir)

    def test_mocap_timestamp(self):
        mocap_timestamp = []
        fbx_file = path_util.get_paths_by_suffix(self.cur_dirs.raw_dir, '.fbx')
        fbx_file_ = open(fbx_file[0], 'r', encoding='ISO-8859-1')
        fbx_content = fbx_file_.readlines()
        fbx_data = fbx_content[1295:]
        i = fbx_content[1294]
        i = i.split(':')[1]
        i = i.strip(' ')
        i = i.split(',')[:-1]
        for time in i:
            mocap_timestamp.append(float(time))
        for i in fbx_data:
            if i != '\t\t} \n':
                i = i.split(',')[:-1]
                for time in i:
                    mocap_timestamp.append(float(time))
            else:
                break
        mocap_timestamp = np.array(mocap_timestamp)
        mocap_timestamp /= 46186158
        mocap_timestamp = mocap_timestamp[1:] - mocap_timestamp[:-1]
        print(f'mocap_timestamp({len(mocap_timestamp)}) offset, min:{mocap_timestamp.min()}|{(mocap_timestamp == mocap_timestamp.min()).sum()}, max:{mocap_timestamp.max()}|{(mocap_timestamp == mocap_timestamp.max()).sum()}, var:{np.var(mocap_timestamp)}')
        if mocap_timestamp.min() == mocap_timestamp.max():
            print("[WARN]mocap时码的每帧间隔均一样!")

    def _gen_pc_and_mocap_indexes(self):
        raise NotImplementedError('当前版本不采用时码进行对齐！')
        pc_start_index = self.seq.to_raw_pcd_index(0)
        pc_end_index = self.seq.to_raw_pcd_index(self.seq.get_pcds_len() - 1)
        mocap_start_index = self.seq.to_raw_mocap_index(0)
        print('gen_pc_and_mocap_indexes:', pc_start_index, mocap_start_index, pc_end_index)

        self.seq.cancel_pcds_map()
        self.seq.cancel_mocaps_map()

        #pc_indexes = np.array([path_util.get_index(filename) for filename in path_util.get_sorted_filenames_by_index(self.cur_dirs.segment_dir, False)])
        pc_indexes = np.arange(pc_start_index, pc_end_index+1)

        mocap_indexes = []
        mocap_timestamp = []
        pc_timestamps = np.array(read_array_dat(os.path.join(self.cur_dirs.pointclouds_dir, 'timestamps.dat')))

        if len(pc_timestamps) != len(self.seq.pcds):
            print('pc_timestamps 和 pcds 数量不匹配！')

        fbx_file = path_util.get_paths_by_suffix(self.cur_dirs.raw_dir, '.fbx')
        fbx_file_ = open(fbx_file[0], 'r')
        fbx_content = fbx_file_.readlines()
        fbx_data = fbx_content[1295:]
        i = fbx_content[1294]
        i = i.split(':')[1]
        i = i.strip(' ')
        i = i.split(',')[:-1]
        for time in i:
            mocap_timestamp.append(float(time))
        for i in fbx_data:
            if i != '\t\t} \n':
                i = i.split(',')[:-1]
                for time in i:
                    mocap_timestamp.append(float(time))
            else:
                break

        mocap_timestamp = [each / 46186158 for each in mocap_timestamp]  # ms

        #cur_process_info = self.get_process_info()
        #start_index = cur_process_info['start_index']
        #pc_end_index = cur_process_info['end_index']
        #pc_start_index = start_index['pointcloud']
        #total_lenth = pc_end_index - pc_start_index + 1
        total_lenth = pc_end_index - pc_start_index + 1

        #mocap_start_index = cur_process_info['start_index']['mocap']

        mocap_start_time = mocap_timestamp[mocap_start_index]
        pc_start_time = pc_timestamps[pc_start_index]

        mocap_indexes.append(mocap_start_index)
        for frame_index in tqdm.tqdm(range(1, total_lenth)):
            # total_lenth是总的pc帧数，其中pc的第一帧和mocap的第一帧已经对上了，接下来找pc剩余的total_length-1个帧对应的mocap帧index
            time_interval = pc_timestamps[pc_start_index + frame_index] - pc_start_time  # s
            mocap_time = mocap_start_time + time_interval * 1000  #
            try:
                a = mocap_timestamp.index(mocap_time)
                mocap_indexes.append(a)
            except:
                temp = mocap_timestamp.copy()
                mocap_timestamp.append(mocap_time)
                b = mocap_timestamp
                b = sorted(b)
                idx = b.index(mocap_time)
                if b[idx] - b[idx - 1] < b[idx + 1] - b[idx]:
                    mocap_indexes.append(idx - 1)
                else:
                    mocap_indexes.append(idx)
                mocap_timestamp = temp.copy()

        mocap_indexes = np.array(mocap_indexes)
        mocap_indexes_path = os.path.join(self.cur_dirs.mocap_dir, 'mocap_indexes.npy')
        np.save(mocap_indexes_path, mocap_indexes)
        self.write_seq_process()
        return pc_indexes, mocap_indexes

    def gen_pc_and_mocap_indexes(self, pc_start_index, pc_end_index, mocap_start_index):
        raise NotImplementedError('当前版本不采用时码进行对齐！')
        #pc_start_index = self.seq.to_raw_pcd_index(0)
        #pc_end_index = self.seq.to_raw_pcd_index(self.seq.get_pcds_len() - 1)
        #mocap_start_index = self.seq.to_raw_mocap_index(0)
        print('gen_pc_and_mocap_indexes:', pc_start_index, mocap_start_index, pc_end_index)

        self.seq.cancel_pcds_map()
        self.seq.cancel_mocaps_map()

        #pc_indexes = np.array([path_util.get_index(filename) for filename in path_util.get_sorted_filenames_by_index(self.cur_dirs.segment_dir, False)])
        pc_indexes = np.arange(pc_start_index, pc_end_index+1)

        mocap_indexes = []
        mocap_timestamp = []
        pc_timestamps = np.array(read_array_dat(os.path.join(self.cur_dirs.pointclouds_dir, 'timestamps.dat')))

        if len(pc_timestamps) != len(self.seq.pcds):
            print('pc_timestamps 和 pcds 数量不匹配！')

        fbx_file = path_util.get_paths_by_suffix(self.cur_dirs.raw_dir, '.fbx')
        fbx_file_ = open(fbx_file[0], 'r')
        fbx_content = fbx_file_.readlines()
        fbx_data = fbx_content[1295:]
        i = fbx_content[1294]
        i = i.split(':')[1]
        i = i.strip(' ')
        i = i.split(',')[:-1]
        for time in i:
            mocap_timestamp.append(float(time))
        for i in fbx_data:
            if i != '\t\t} \n':
                i = i.split(',')[:-1]
                for time in i:
                    mocap_timestamp.append(float(time))
            else:
                break

        mocap_timestamp = [each / 46186158 for each in mocap_timestamp]  # ms

        total_lenth = pc_end_index - pc_start_index + 1

        #mocap_start_index = cur_process_info['start_index']['mocap']

        mocap_start_time = mocap_timestamp[mocap_start_index]
        pc_start_time = pc_timestamps[pc_start_index]
        #"""
        mocap_seq_t = np.array(mocap_timestamp[mocap_start_index:])
        gap = mocap_seq_t[1:] - mocap_seq_t[:-1]
        print(f'mocap timestamp gap:{np.unique(gap)}')
        print(list(zip(np.where(gap > 10)[0] + mocap_start_index, gap[gap > 10])))
        """
        gap[gap > 30] = 30
        mocap_seq_t = [mocap_start_time, ]
        for e in gap:
            mocap_seq_t.append(mocap_seq_t[-1] + e)
        mocap_timestamp = mocap_timestamp[:mocap_start_index] + mocap_seq_t#"""

        mocap_indexes.append(mocap_start_index)
        for frame_index in tqdm.tqdm(range(1, total_lenth)):
            # total_lenth是总的pc帧数，其中pc的第一帧和mocap的第一帧已经对上了，接下来找pc剩余的total_length-1个帧对应的mocap帧index
            time_interval = pc_timestamps[pc_start_index + frame_index] - pc_start_time  # s
            mocap_time = mocap_start_time + time_interval * 1000  #
            try:
                a = mocap_timestamp.index(mocap_time)
                mocap_indexes.append(a)
            except:
                temp = mocap_timestamp.copy()
                temp.append(mocap_time)
                b = temp
                b = sorted(b)
                idx = b.index(mocap_time)
                if b[idx] - b[idx - 1] < b[idx + 1] - b[idx]:
                    mocap_indexes.append(idx - 1)
                else:
                    mocap_indexes.append(idx)

        mocap_indexes = np.array(mocap_indexes)
        mocap_indexes_path = os.path.join(self.cur_dirs.mocap_dir, 'mocap_indexes.npy')
        np.save(mocap_indexes_path, mocap_indexes)

        return pc_indexes, mocap_indexes

    def quick_gen_pc_and_mocap_indexes(self, pc_start_index, pc_end_index, mocap_start_index):
        #pc_start_index = self.seq.to_raw_pcd_index(0)
        #pc_end_index = self.seq.to_raw_pcd_index(self.seq.get_pcds_len() - 1)
        #mocap_start_index = self.seq.to_raw_mocap_index(0)
        print('gen_pc_and_mocap_indexes:', pc_start_index, mocap_start_index, pc_end_index)

        self.seq.cancel_pcds_map()
        self.seq.cancel_mocaps_map()

        max_pc_end_index = (len(self.seq.mocaps) - 1 - mocap_start_index) // 10 + pc_start_index
        max_pc_end_index = min(len(s.seq.pcds)-1, max_pc_end_index)
        if pc_end_index > max_pc_end_index:
            print(f'最大pc_end_index为：{max_pc_end_index}!')
            pc_end_index = max_pc_end_index

        #pc_indexes = np.array([path_util.get_index(filename) for filename in path_util.get_sorted_filenames_by_index(self.cur_dirs.segment_dir, False)])
        pc_indexes = np.arange(pc_start_index, pc_end_index+1)

        mocap_end_index = mocap_start_index + (pc_end_index - pc_start_index) * 10
        mocap_indexes = np.arange(mocap_start_index, mocap_end_index+1, 10)

        assert mocap_end_index < len(self.seq.mocaps), f'mocap_end_index:{mocap_end_index}超过mocaps范围！'

        mocap_indexes_path = os.path.join(self.cur_dirs.mocap_dir, 'mocap_indexes.npy')
        np.save(mocap_indexes_path, mocap_indexes)

        return pc_indexes, mocap_indexes

    def _get_smpl_vertices(self, poses):
        from smpl import model as smpl_model
        n_poses = len(poses)
        batch_size = 512
        n_batch = (n_poses + batch_size - 1) // batch_size
        smpl = smpl_model.SMPL()
        vertices = np.zeros((0, 6890, 3))
        joints = np.zeros((0, 24, 3))
        for i in tqdm.tqdm(range(n_batch), desc='Gen Smpl Vertices', ncols=60):
            lb = i * batch_size
            ub = min((i + 1) * batch_size, n_poses)
            cur_poses = np.stack(poses[lb:ub])
            cur_vertices = smpl.get_vertices(pose=cur_poses)
            cur_joints = smpl.get_joints(cur_vertices)
            vertices = np.concatenate((vertices, cur_vertices))
            joints = np.concatenate((joints, cur_joints))

        return vertices, joints

    def get_smpl_vertices(self, poses):
        n_poses = len(poses)
        batch_size = 16

        batches = [np.stack(poses[l:r]) for l, r in
                   zip(range(0, n_poses + batch_size, batch_size), range(batch_size, n_poses + batch_size, batch_size))]
        vertices = multiprocess.multi_func(
            self.smpl.get_vertices, min(MAX_PROCESS_COUNT, len(batches)), len(batches), 'get smpl vertices:', True, batches)
        vertices = np.concatenate(vertices, axis=0)
        #cur_joints = smpl.get_joints(cur_vertices)

        return vertices

    def get_smpl_joints_from_vertices(self, vertices):
        n_vertices = len(vertices)
        batch_size = 16

        batches = [np.stack(vertices[l:r]) for l, r in
                   zip(range(0, n_vertices + batch_size, batch_size), range(batch_size, n_vertices + batch_size, batch_size))]
        joints = multiprocess.multi_func(
            self.smpl.get_joints, min(MAX_PROCESS_COUNT, len(batches)), len(batches), 'get smpl joints:', True, batches)
        joints = np.concatenate(joints, axis=0)
        return joints

    def get_mocap_to_lidar_trans(self, poses, segment_pcds, lidar_to_mocap_RT):
        mocap_point_clouds = self.get_smpl_vertices(poses)

        assert len(poses) == len(segment_pcds)
        n = len(segment_pcds)
        #poses = [mocap_data.pose(mocap_index)for mocap_index in tqdm(mocap_indexes)]

        mocap_to_lidar_translations = multiprocess.multi_func(
            transformation.get_mocap_to_lidar_translation, MAX_PROCESS_COUNT, len(segment_pcds), 'calculate trans::', True, mocap_point_clouds, segment_pcds, [lidar_to_mocap_RT, ] * len(segment_pcds))

        #mocap_to_lidar_translations = [transformation.get_mocap_to_lidar_translation(
            #mocap_points, segment_points[:, :3], lidar_to_mocap_RT) for mocap_points, segment_points in
            #tqdm.tqdm(zip(mocap_point_clouds, segment_pcds), total=n)]

        # 平滑平移量
        half_width = 10
        translation_sum = np.zeros((3,))
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

        return mocap_to_lidar_translations

    def get_quick_mocap_to_lidar_trans(self, poses, segment_pcds, lidar_to_mocap_RT):
        print('Calculate MoCap to LiDAR translations:')

        mocap_to_lidar_translations = [pcd.mean(axis=0) for pcd in segment_pcds]

        return mocap_to_lidar_translations

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

    def _generate_pose(self):
        from process import save_smpl_data
        from smpl import model as smpl_model
        assert self.seq.get_mocaps_len() == self.seq.get_pcds_len() == len(self.seq.get_cur_pcds_map())
        poses = [self.seq.mocaps.pose(i) for i in tqdm.tqdm(self.seq.get_cur_mocap_map())]
        segment_filenames = [f'{i}.pcd' for i in self.seq.get_cur_pcds_map()]
        mocap_to_lidar_translations = [self.seq.get_mocap_transform(i)[:3, 3] for i in self.seq.get_cur_mocap_map()]
        mocap_point_clouds, mocap_joints = self.get_smpl_vertices(poses)
        beta = np.zeros((10,))
        lidar_to_mocap_RT = self.seq.get_cur_transform()
        smpl = smpl_model.SMPL()
        for mocap_points, cur_translation, pose, seg_filename in tqdm.tqdm(
                zip(mocap_point_clouds, mocap_to_lidar_translations, poses, segment_filenames), total=len(poses)):
            save_smpl_data(mocap_points, cur_translation, pose, seg_filename,
                           beta, lidar_to_mocap_RT, self.cur_dirs.pose_dir, smpl)
        self.write_process_info()
        print(f"Success generate pose, from:{self.seq.to_raw_pcd_index(0)} to:{self.seq.to_raw_pcd_index(self.seq.get_pcds_len()-1)}, total:{self.seq.get_pcds_len()}")

    def generate_pose(self):
        from scipy.spatial.transform import Rotation as R
        assert self.seq.get_mocaps_len() == self.seq.get_pcds_len() == len(self.seq.get_cur_pcds_map())

        #poses = [self.seq.mocaps.pose(i) for i in tqdm.tqdm(self.seq.get_cur_mocap_map())]
        poses = self.seq.get_cur_mocap_list()

        lidar_to_mocap_RT = self.seq.get_cur_transform()
        mocap_to_lidar_RT = np.linalg.inv(lidar_to_mocap_RT)
        mocap_to_lidar_R = mocap_to_lidar_RT[:3, :3]
        poses_r = []
        for pose in poses:
            pose_r = pose.copy()
            pose_r[:3] = (R.from_matrix(mocap_to_lidar_R) * R.from_rotvec(pose[0:3])).as_rotvec()
            poses_r.append(pose_r)

        filename_indexes = [i for i in self.seq.get_cur_pcds_map()]

        mocap_to_lidar_translations = [self.seq.get_mocap_transform(i)[:3, 3] for i in self.seq.get_cur_mocap_map()]

        mocap_point_clouds = self.get_smpl_vertices(poses)

        mocap_point_clouds_r = self.get_smpl_vertices(poses_r)

        mocap_point_clouds_rt = [transformation.mocap_to_lidar(mocap_points, lidar_to_mocap_RT, translation=cur_translation) for mocap_points, cur_translation in zip(mocap_point_clouds, mocap_to_lidar_translations)]

        mocap_to_lidar_translations = [mocap_pc_rt[0] - mocap_pc_r[0] for mocap_pc_rt, mocap_pc_r in zip(mocap_point_clouds_rt, mocap_point_clouds_r)]

        filenames = [os.path.join(self.cur_dirs.pose_dir, '{}.ply'.format(filename_index)) for filename_index in filename_indexes]
        multiprocess.multi_func(save_ply, MAX_PROCESS_COUNT, len(filenames), 'save pose ply:', True, mocap_point_clouds, filenames)

        filenames = [os.path.join(self.cur_dirs.pose_dir, '{}.json'.format(filename_index)) for filename_index in filename_indexes]
        multiprocess.multi_func(save_smpl_json, MAX_PROCESS_COUNT, len(filenames), 'save pose json:', True, filenames, poses_r, mocap_to_lidar_translations)

        """
        for mocap_pc_rt, trans, pose, filename_index in zip(mocap_point_clouds, mocap_to_lidar_translations, poses, filename_indexes):
            save_ply(mocap_pc_rt, os.path.join(self.cur_dirs.pose_dir, '{}.ply'.format(filename_index)))
            save_smpl_json(os.path.join(self.cur_dirs.pose_dir, '{}.json'.format(filename_index)), pose, trans)"""

        self.write_process_info()
        print(f"Success generate pose, from:{self.seq.to_raw_pcd_index(0)} to:{self.seq.to_raw_pcd_index(self.seq.get_pcds_len()-1)}, total:{self.seq.get_pcds_len()}")

    def generate_smpl(self):
        from process import generate_smpl
        generate_smpl(self.cur_dirs, self.seq.mocaps, self.seq.get_cur_mocap_map(), self.get_process_info())
        print("Success generate smpl")

    def generate_segment(self):
        import open3d as o3d

        segment_dir = self.cur_dirs.segment_dir
        for i in tqdm.tqdm(range(self.seq.get_pcds_len()), desc='Gen Segment', ncols=60):
            pcd = self.seq.get_cur_pcd_with_no_rt(i)
            filename = os.path.join(segment_dir, f'{self.seq.to_raw_pcd_index(i)}.pcd')
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(pcd)
            o3d.io.write_point_cloud(filename, pointcloud)
        print(f"Success generate segment, from:{self.seq.to_raw_pcd_index(0)} to:{self. seq.to_raw_pcd_index(self.seq.get_pcds_len()-1)}, total:{self.seq.get_pcds_len()}")

    def generate_dataset(self):
        from scipy.spatial.transform import Rotation as R
        assert self.seq.get_mocaps_len() == self.seq.get_pcds_len() == len(self.seq.get_cur_pcds_map())

        # poses = [self.seq.mocaps.pose(i) for i in tqdm.tqdm(self.seq.get_cur_mocap_map())]
        poses = self.seq.get_cur_mocap_list()

        lidar_to_mocap_RT = self.seq.get_cur_transform()
        mocap_to_lidar_RT = np.linalg.inv(lidar_to_mocap_RT)
        mocap_to_lidar_R = mocap_to_lidar_RT[:3, :3]
        poses_r = []
        for pose in poses:
            pose_r = pose.copy()
            pose_r[:3] = (R.from_matrix(mocap_to_lidar_R) * R.from_rotvec(pose[0:3])).as_rotvec()
            poses_r.append(pose_r)

        mocap_to_lidar_translations = [self.seq.get_mocap_transform(i)[:3, 3] for i in self.seq.get_cur_mocap_map()]

        mocap_point_clouds = self.get_smpl_vertices(poses)

        mocap_point_clouds_r = self.get_smpl_vertices(poses_r)

        mocap_point_clouds_rt = [
            transformation.mocap_to_lidar(mocap_points, lidar_to_mocap_RT, translation=cur_translation) for
            mocap_points, cur_translation in zip(mocap_point_clouds, mocap_to_lidar_translations)]

        mocap_to_lidar_translations = [mocap_pc_rt[0] - mocap_pc_r[0] for mocap_pc_rt, mocap_pc_r in
                                       zip(mocap_point_clouds_rt, mocap_point_clouds_r)]

        whole_poses = np.stack(poses_r)
        whole_betas = np.zeros((len(whole_poses), 10))
        whole_trans = mocap_to_lidar_translations
        whole_lidar_to_mocap_RT = self.seq.get_cur_transform()[np.newaxis, ...].repeat(len(whole_poses), axis=0)
        whole_rotmats = R.from_rotvec(whole_poses.reshape(-1, 3)).as_matrix().reshape(len(whole_poses), 24, 3, 3)
        whole_full_joints = self.get_smpl_joints_from_vertices(mocap_point_clouds_r)
        pcds = [self.seq.get_cur_pcd_with_no_rt(i) for i in
                        tqdm.tqdm(range(self.seq.get_pcds_len()), desc='Gen Segment', ncols=60)]
        person_label_list = [self.seq.person_label_list[i] for i in self.seq.get_cur_pcds_map()]
        whole_point_clouds = []
        whole_body_label = []
        whole_points_nums = []

        for label, pc in zip(person_label_list, pcds):
            whole_points_nums.append(len(pc))
            fixed = fix_points_num(np.concatenate((pc, label[:, np.newaxis]), axis=-1), 512)
            whole_point_clouds.append(fixed[:, :3])
            whole_body_label.append(fixed[:, 3])

        extras_path = '/SAMSUMG8T/mqh/lidarcapv2/dataset'
        whole_filename = f'{self.index}.hdf5'
        with h5py.File(os.path.join(extras_path, whole_filename), 'w') as f:
            f.create_dataset('pose', data=whole_poses)
            f.create_dataset('shape', data=whole_betas)
            f.create_dataset('trans', data=whole_trans)
            f.create_dataset('point_clouds', data=whole_point_clouds)
            f.create_dataset('points_num', data=whole_points_nums)
            f.create_dataset('full_joints', data=whole_full_joints)
            f.create_dataset('lidar_to_mocap_RT', data=whole_lidar_to_mocap_RT)
            f.create_dataset('rotmats', data=whole_rotmats)
            f.create_dataset('body_label', data=whole_body_label)
        print(f'success generate dataset:{h5py.File(os.path.join(extras_path, whole_filename))}')
        self.write_seq_process()
        self.write_process_info()

    def vis_pose(self, index):
        index = self.seq.to_raw_pcd_index(index)
        with open(os.path.join(self.cur_dirs.pose_dir, f'{index}.json')) as f:
            pose = json.load(f)
            return pose['pose'], pose['trans']

    def vis_segment(self, index):
        index = self.seq.to_raw_pcd_index(index)
        import open3d as o3d
        segment_dir = self.cur_dirs.segment_dir
        filename = os.path.join(segment_dir, f'{index}.pcd')
        return np.array(o3d.io.read_point_cloud(filename).points).tolist()

#np.loadtxt(path_util.get_paths_by_suffix(self.cur_dirs.raw_dir, '.txt')[0], dtype=str, delimiter='\t',skiprows=1)[:, -10:-7].astype(float)
#np.array(o3d.io.read_point_cloud(path_util.get_paths_by_suffix(s.cur_dirs.raw_dir, '.ply')[0]).points)
async def vis_update():
    while(True):
        await asyncio.sleep(0.01)

sio = socketio.AsyncServer(async_mode='tornado')


s = server()
##s.init('/SAMSUMG8T/mqh/lidarcapv2/raw', '/SAMSUMG8T/mqh/lidarcapv2/lidarcap', 51802)
#s.load_process_info()

@sio.event
def init(sid, raw_dir, dataset_dir, index):
    return s.init(raw_dir, dataset_dir, index)

@sio.event
def cancel_transform(sid):
    return s.seq.cancel_transform()

@sio.event
def apply_transform(sid, T):
    if isinstance(T, list):
        return s.seq.apply_transform(np.array(T).reshape(4, 4))
    elif isinstance(T, int) or isinstance(T, str):
        return s.apply_transform_from_process_info(T)
    else:
        print(f'ERROR:apply_transform:非法的参数：{T}')
    s.write_process_info()

@sio.event
def apply_voxel_mask(sid, voxel_size, t_threshold, a):
    return s.seq.apply_voxel_mask(voxel_size, t_threshold, a)

@sio.event
def apply_ground_mask(sid, index_l, index_r, z_min, z_max):
    ret = s.seq.apply_ground_mask(index_l, index_r, z_min, z_max)
    s.write_seq_process()
    return ret

@sio.event
def apply_box_mask(sid, index_l, index_r, x_limit, y_limilt, inverse=False):
    ret = s.seq.apply_box_mask(index_l, index_r, x_limit, y_limilt, inverse)
    s.write_seq_process()
    return ret

@sio.event
def apply_trace_and_cluster_mask(sid, index_l, index_r, start_pos, radius, eps):
    ret = s.seq.apply_trace_and_cluster_mask(index_l, index_r, start_pos, radius, eps)
    s.write_seq_process()
    return ret


@sio.event
def apply_cluster_mask(sid, index, start_pos, radius, eps):
    ret = s.seq.apply_cluster_mask(index, start_pos, radius, eps)
    s.write_seq_process()
    return ret

@sio.event
def cut_out_person(sid, index_l, index_r, start_pos, radius):
    index_s = s.seq.to_raw_pcd_index(0)
    assert index_l >= index_s, '找不到对应的pose!'
    poses = [np.array(s.seq.get_cur_mocap(s.seq.raw_pcd_index_to_cur(i))[0]) for i in tqdm.tqdm(range(index_l, index_r+1, 1), desc='get cur mocap list')]
    mocap_point_clouds = s.get_smpl_vertices(poses)

    ret = s.seq.cut_out_person(index_l, index_r, start_pos, radius, mocap_point_clouds)
    s.write_seq_process()
    return ret

@sio.event
def icp_person(sid, index_l, index_r, trans_move_limit, dist_threshold, use_cur_trans, fix_trans, use_mocap_trans, fix_pc):
    index_s = s.seq.to_raw_pcd_index(0)
    assert index_l >= index_s, '找不到对应的pose!'
    poses = [np.array(s.seq.get_cur_mocap(s.seq.raw_pcd_index_to_cur(i))[0]) for i in tqdm.tqdm(range(index_l, index_r+1, 1), desc='get cur mocap list')]
    mocap_point_clouds = s.get_smpl_vertices(poses)

    if use_cur_trans:
        raise NotImplementedError('此模块暂时不可用！')
        ret = s.seq.icp_person_finetune(index_l, index_r, trans_move_limit, dist_threshold, mocap_point_clouds, fix_trans=fix_trans)
    else:
        ret = s.seq.icp_person(index_l, index_r, trans_move_limit, dist_threshold, mocap_point_clouds, use_cur_trans=use_cur_trans, fix_trans=fix_trans, use_mocap_trans=use_mocap_trans, fix_pc=fix_pc)
    s.write_seq_process()
    return ret

@sio.event
def icp_person_per_frame(sid, frame, center_pos):
    index_s = s.seq.to_raw_pcd_index(0)
    assert frame >= index_s, '找不到对应的pose!'
    poses = [np.array(s.seq.get_cur_mocap(s.seq.raw_pcd_index_to_cur(i))[0]) for i in tqdm.tqdm(range(frame, frame+1, 1), desc='get cur mocap list')]
    mocap_point_clouds = s.get_smpl_vertices(poses)

    s.seq.icp_person_per_frame(frame, center_pos, mocap_point_clouds[0])
    print('icp person per frame finish!')

@sio.event
def icp_object(sid, index_l, index_r, trans_limit, rotvec_limit):
    ret = s.seq.icp_object(index_l, index_r, trans_limit=trans_limit, rotvec_limit=rotvec_limit)
    s.write_seq_process()
    return ret

@sio.event
def icp_object_per_frame(sid, frame):
    s.seq.icp_object_per_frame(frame)
    print('icp object per frame finish!')

@sio.event
def load_object(sid, name):
    s.seq.load_obj(name)
    print('success load object!')

@sio.event
def cancel_mask(sid):
    return s.seq.cancel_mask()

@sio.event
def get_cur_pcd(sid, index, raw_index=False):
    pcd = s.seq.get_cur_pcd(index, raw_index=raw_index)
    if len(pcd) > 16000:
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(pcd)
        pointcloud = pointcloud.voxel_down_sample(0.25)
        pcd = np.array(pointcloud.points)
    return pcd.tolist()

@sio.event
def get_cur_pcd_and_label(sid, index, raw_index=False):
    pcd, label = s.seq.get_cur_pcd_and_label(index, raw_index=raw_index)
    if len(pcd) > 16000:
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(pcd)
        pointcloud = pointcloud.voxel_down_sample(0.25)
        pcd = np.array(pointcloud.points)
    return pcd.tolist(), label.tolist() if label is not None else label

@sio.event
def get_pcds_len(sid):
    return s.seq.get_pcds_len()

@sio.event
def get_cur_mocap(sid, index, raw_index=False):
    return s.seq.get_cur_mocap(index, raw_index=raw_index)

@sio.event
def get_cur_pcd_and_mocap(sid, index):
    #pcd = (s.seq.to_raw_pcd_index(index), get_cur_pcd(None, index))
    pcd = (s.seq.to_raw_pcd_index(index), get_cur_pcd_and_label(None, index))
    mocap = (s.seq.to_raw_mocap_index(index), get_cur_mocap(None, index))
    obj_rt = s.seq.get_obj_rt(s.seq.to_raw_pcd_index(index))
    return pcd, mocap, obj_rt.tolist() if obj_rt is not None else obj_rt

@sio.event
def get_mocaps_len(sid):
    return s.seq.get_mocaps_len()

@sio.event
def apply_mocaps_map_to_pcds(sid, a, b, c):
    #pcds_map, mocaps_map = s.gen_pc_and_mocap_indexes(a, b, c)
    pcds_map, mocaps_map = s.quick_gen_pc_and_mocap_indexes(a, b, c)
    s.seq.cancel_pcds_map()
    s.seq.cancel_mocaps_map()
    s.seq.apply_pcds_map(pcds_map)
    s.seq.apply_mocaps_map(mocaps_map)
    #s.write_seq_process()
    s.write_process_info()
    print('success gen index!')

@sio.event
def cancel_mocaps_map_to_pcds(sid):
    s.seq.cancel_pcds_map()
    s.seq.cancel_mocaps_map()
    print('success cancel index!')

@sio.event
def apply_trans_to_mocaps(sid):
    assert s.seq.get_mocaps_len() == s.seq.get_pcds_len()
    poses = s.seq.get_cur_mocap_list()
    #segment_pcds = [s.seq.get_cur_pcd(i) for i in tqdm.tqdm(range(s.seq.get_pcds_len()), desc='get cur pcds')]
    segment_pcds = s.seq.get_cur_pcd_list()
    mocap_to_lidar_trans = s.get_mocap_to_lidar_trans(poses, segment_pcds, np.identity(4))
    rt_list = np.identity(4)[np.newaxis, ...].repeat(len(mocap_to_lidar_trans), axis=0)
    for rt, trans in zip(rt_list, mocap_to_lidar_trans):
        rt[:3, 3] = trans
    s.seq.apply_mocap_transform(s.seq.get_cur_mocap_map(), rt_list)
    s.write_seq_process()

@sio.event
def apply_quick_trans_to_mocaps(sid):
    assert s.seq.get_mocaps_len() == s.seq.get_pcds_len()
    #poses = [s.seq.mocaps.pose(i) for i in tqdm.tqdm(s.seq.get_cur_mocap_map())]

    #segment_pcds = [s.seq.get_cur_pcd(i) for i in tqdm.tqdm(range(s.seq.get_pcds_len()), desc='get cur pcds')]
    segment_pcds = s.seq.get_cur_pcd_list()


    mocap_to_lidar_trans = s.get_quick_mocap_to_lidar_trans(None, segment_pcds, np.identity(4))
    rt_list = np.identity(4)[np.newaxis, ...].repeat(len(mocap_to_lidar_trans), axis=0)
    for rt, trans in zip(rt_list, mocap_to_lidar_trans):
        rt[:3, 3] = trans
    s.seq.apply_mocap_transform(s.seq.get_cur_mocap_map(), rt_list)
    s.write_seq_process()

@sio.event
def access_seq(sid, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    return s.seq.access(func, args, kwargs)

@sio.event
def access_server(sid, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    return eval(f's.{func}(*args, **kwargs)')


setting = {
    'clips': []  #其中的元素是四元组
}

"""
@sio.event
def get_setting(sid):
    return setting

@sio.event
def apply_setting(sid, msg):
    updated_clips = []
    for new_clip, clip in zip(msg['clips'], setting['clips']):
        if new_clip != clip:
            pcds_map, mocaps_map = s.gen_pc_and_mocap_indexes(new_clip[0], new_clip[1], new_clip[2])
            s.seq.apply_pcds_map(pcds_map)
            s.seq.apply_mocaps_map(mocaps_map)
            new_clip = (pcds_map[0], pcds_map[-1], mocaps_map[0], mocaps_map[-1])
            s.write_seq_process()
        updated_clips.append(new_clip)

    setting['clips'] = updated_clips
    return get_setting()"""


def run(port, index):
    s.init('/SAMSUMG8T/mqh/lidarcapv2/raw', '/SAMSUMG8T/mqh/lidarcapv2/lidarcap', index)
    #cut_out_person(None, 21, 3000, 0, 1)
    app = tornado.web.Application(
        [
            (r"/socket.io/", socketio.get_tornado_handler(sio)),
        ],
        # ... other application options
    )
    app.listen(port)
    print(f'server listen on:{port}')
    tornado.ioloop.IOLoop.current().run_sync(vis_update)


"""
s.init('/SAMSUMG8T/mqh/lidarcapv2/raw', '/SAMSUMG8T/mqh/lidarcapv2/lidarcap', 1)
s.load_process_info()
import socketio
client = socketio.Client()
client.connect('http://172.18.69.139:5555')
i  += 1
pc = s.vis_segment(i)
pose, trans = s.vis_pose(i)

pc1 = o3d.io.read_point_cloud(os.path.join(s.cur_dirs.pose_dir, f'{i}.ply'))
pc1 = np.array(pc1.points).tolist()
client.emit('add_pc', ('pc', pc))
client.emit('add_pc', ('pc1', pc1))
client.emit('add_smpl_pc', ('human_pc', pose, None, trans))
"""