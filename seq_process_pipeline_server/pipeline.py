import numpy as np
import open3d as o3d
import os
from functools import reduce
import tqdm
import json

#import sys
#sys.path.append(sys.path.append('/cwang/home/ljl/anaconda3/envs/torch/lib/python3.7/site-packages/python_pcl-0.3.0rc1-py3.7-linux-x86_64.egg'))
from seq_process_pipeline_server import pipeline_util
from seq_process_pipeline_server.pipeline_util import read_array_dat, affine
from util import path_util, mocap_util, transformation, multiprocess

import torch
torch.set_num_threads(1)
import pcl
from scipy.spatial.transform import Rotation as R

# TODO: 显示的人头顶显示当前帧数


#   定义一个序列类，其中包含数据处理的各种算法（切分人体，配准等），输入是我们采集的动捕序列和点云，输出是数据集

def mqh_sa_icp_wo_r(pc, v, t_init, trans_move_limit, dist_threshold, z_init):
    t, m = transformation.mqh_sa_icp_wo_r(v, pc, t_init, trans_move_limit=trans_move_limit, T=1, Tmin=0.1,
                                          dist_threshold=dist_threshold,
                                          k=5, z=z_init)
    t = -t

    return m, t

class seq_process:
    def __init__(self, pcds, mocaps):
        # mask和transform和map三种操作是独立的，均可独立apply或者cancel,相互之间没有依赖关系
        self.pcds = None
        self.mocaps = None

        self.pcds_transform_list = []
        self.pcds_mask_list = []
        self.pcds_map_list = []
        self.mocaps_transform = {}
        self.mocaps_map_list = []

        self.person_label_list = []
        self.object_rt_list = []
        self.obj = None

        if pcds is not None:
            self.load_pcds(pcds)
        if mocaps is not None:
            self.load_mocaps(mocaps)

        self.mocap_pose_cache = {}

    def __getstate__(self):
        state = {"pcds_transform_list": self.pcds_transform_list,
                 "pcds_mask_list": self.pcds_mask_list,
                 "pcds_map_list": self.pcds_map_list,
                 "mocaps_transform": self.mocaps_transform,
                 "mocaps_map_list": self.mocaps_map_list,
                 'person_label_list': self.person_label_list,
                 'object_rt_list': self.object_rt_list,
                 'mocap_pose_cache': self.mocap_pose_cache,
                 'obj': self.obj
                 }
        return state

    def load_obj(self, name):
        self.obj = pipeline_util.read_obj(f'/SAMSUMG8T/mqh/lidarcapv2/raw/object/{name}.obj')

    def load_pcds(self, pcds):
        self.pcds = pcds
        self.pcds_map_list.append(np.arange(len(self.pcds)))

        self.pcds_transform_list = []
        self.pcds_mask_list = [[np.ones((self.pcds[i].shape[0],)) == 0 for i in range(len(self.pcds))], ]

        self.person_label_list = [None for i in range(len(pcds))]

    def load_mocaps(self, mocaps):
        self.mocaps = mocaps
        self.mocaps_map_list.append(np.arange(len(self.mocaps)))

        self.mocaps_transform = {}

    def apply_transform(self, transform_matrix):
        self.pcds_transform_list.append(transform_matrix)

    def cancel_transform(self):
        if self.pcds_transform_list:
            self.pcds_transform_list.pop()
            return True
        else:
            return False

    def get_cur_transform(self):
        return reduce(lambda rot1, rot2: rot2.dot(rot1), self.pcds_transform_list) if self.pcds_transform_list else np.eye(4)

    def apply_mocap_transform(self, indexes, transform_matrix):
        for index, transf in zip(indexes, transform_matrix):
            self.mocaps_transform[index] = np.array(transf).reshape(4, 4)

    def cancel_mocap_transform(self):
        self.mocaps_transform = {}

    def get_mocap_transform(self, index):
        return self.mocaps_transform[index] if index in self.mocaps_transform else np.identity(4)

    def apply_voxel_mask(self, voxel_size, t_threshold, a):
        pcds = self.pcds
        VOXEL_SIZE = voxel_size
        MIN = np.stack([np.amin(pcd, axis=0) for pcd in pcds], axis=0).min(axis=0, keepdims=True)
        MAX = np.stack([np.amax(pcd, axis=0) for pcd in pcds], axis=0).max(axis=0, keepdims=True)
        VOXEL_COUNT = (MAX - MIN) / VOXEL_SIZE + 1
        #TOTAL_VOXEL_COUNT = int((VOXEL_COUNT[0, 0] + 1) * (VOXEL_COUNT[0, 1] + 1) * (VOXEL_COUNT[0, 2] + 1))
        TOTAL_VOXEL_COUNT = int(VOXEL_COUNT[0, 0] * VOXEL_COUNT[0, 1]* VOXEL_COUNT[0, 2])

        masks = []
        pc_masks = []
        for i in range(len(pcds)):
            print(i)
            voxel_ids = (np.array([1, VOXEL_COUNT[0, 0], VOXEL_COUNT[0, 0] * VOXEL_COUNT[0, 1]]).dot(
                ((pcds[i] - MIN) / voxel_size).T)).astype('int32')
            mask = np.zeros((TOTAL_VOXEL_COUNT,), dtype='bool')  # 用bool运行更快
            mask[voxel_ids] = True
            # mask = np.concatenate((mask, np.array([TOTAL_VOXEL_COUNT - 1, ])))
            # mask = np.bincount(mask) > 0

            if i < t_threshold:
                masks.append(mask)
            else:
                assert len(masks) == t_threshold
                # point_mask = np.arange(TOTAL_VOXEL_COUNT)[np.stack(masks, axis=0).sum(axis=0) > 0]
                voxel_mask = np.stack(masks, axis=0).sum(axis=0)
                voxel_mask = np.logical_or(voxel_mask <= a, voxel_mask >= t_threshold - a)
                if i == t_threshold:
                    for j in range(t_threshold - (t_threshold // 2)):
                        voxel_ids = (np.array([1, VOXEL_COUNT[0, 0], VOXEL_COUNT[0, 0] * VOXEL_COUNT[0, 1]]).dot(
                            ((pcds[j] - MIN) / voxel_size).T)).astype('int32')
                        pc_mask = voxel_mask[voxel_ids]
                        pc_masks.append(pc_mask)

                voxel_ids = (np.array([1, VOXEL_COUNT[0, 0], VOXEL_COUNT[0, 0] * VOXEL_COUNT[0, 1]]).dot(
                    ((pcds[i - (t_threshold // 2)] - MIN) / voxel_size).T)).astype('int32')
                pc_mask = voxel_mask[voxel_ids]
                pc_masks.append(pc_mask)
                masks.pop(0)
                masks.append(mask)

                if i == len(pcds) - 1:
                    for j in range(len(pcds) - (t_threshold // 2), len(pcds)):
                        voxel_ids = (np.array([1, VOXEL_COUNT[0, 0], VOXEL_COUNT[0, 0] * VOXEL_COUNT[0, 1]]).dot(
                            ((pcds[j] - MIN) / voxel_size).T)).astype('int32')
                        pc_mask = voxel_mask[voxel_ids]
                        pc_masks.append(pc_mask)

        self.apply_mask(pc_masks)

    def apply_ground_mask(self, index_l, index_r, z_min, z_max):
        masks = []
        for i in tqdm.tqdm(range(index_l, index_r+1, 1), desc='apply_ground_mask', ncols=60):
            pcd = self.get_cur_pcd(i, raw_index=True)
            z = pcd[:, 2]
            mask = np.logical_and(z > z_min, z < z_max)
            masks.append(mask)
        masks = [True, ] * index_l + masks + \
                [True, ] * (len(self.pcds) - index_r - 1)
        self.apply_mask(masks)

    def apply_box_mask(self, index_l, index_r, x_limit, y_limit, inverse):
        masks = []
        for i in tqdm.tqdm(range(index_l, index_r+1, 1), desc='apply_box_mask', ncols=60):
            pcd = self.get_cur_pcd(i, raw_index=True)
            x, y = pcd[:, 0], pcd[:, 1]
            mask = np.logical_and(np.logical_and(x > x_limit[0], x < x_limit[1]), np.logical_and(y > y_limit[0], y < y_limit[1]))
            masks.append(mask if not inverse else np.logical_not(mask))
        masks = [True, ] * index_l + masks + \
                [True, ] * (len(self.pcds) - index_r - 1)
        self.apply_mask(masks)

    def cut_out_person(self, index_l, index_r, start_pos, radius, mocap_point_clouds):
        # 通过mocap_points和ball query后点云进行ICP来获取人体中心位置,中心位置再进行一次ball query即可得到人体点云（不需要聚类）
        t_list = []
        masks = []
        center_pos = start_pos

        for frame, v in tqdm.tqdm(zip(range(index_l, index_r+1, 1), mocap_point_clouds), desc='cut_out_person', ncols=60):
            pc = self.get_cur_pcd(frame, raw_index=True)
            mask1 = ((pc - center_pos) ** 2).sum(axis=1) < radius ** 2
            p = pc[mask1]

            if len(p) == 0:
                print(f'cut_out_person:lose person at frame:{frame}, break!')
                index_r = frame-1
                break

            if frame == index_l:
                t = -transformation.icp_without_rotation(p.astype(np.float32), mocap_point_clouds[0].astype(np.float32),
                                                         pcl.PointCloud(
                                                             mocap_point_clouds[0].astype(np.float32)).make_kdtree_flann())
                z = t[2]
                print(f'cut out person from index:{index_l} to index:{index_r}, radius:{radius}, z:{z}')


            t, m = transformation.mqh_sa_icp_wo_r(v, p, -t, trans_move_limit=0.2, T=1, Tmin=0.1, dist_threshold=0.01,
                                                  k=5, z=-z)  # 611107开头效果很好
            t = -t
            #t = transformation.icp_without_rotation(v, pcds[frame], pcl.PointCloud(pcds[frame].astype(np.float32)).make_kdtree_flann())

            center_pos = p[m].mean(axis=0)
            mask2 = ((p - center_pos) ** 2).sum(axis=1) < radius ** 2
            masks.append(self.mask_seq_and(np.logical_not(mask1), np.logical_not(mask2)))

            t_list.append(t)


        rt_list = np.identity(4)[np.newaxis, ...].repeat(len(mocap_point_clouds), axis=0)
        for rt, trans in zip(rt_list, t_list):
            rt[:3, 3] = trans
        indexes = [self.to_raw_mocap_index(self.raw_pcd_index_to_cur(i)) for i in range(index_l, index_r+1, 1)]
        self.apply_mocap_transform(indexes, rt_list)

        masks = [True, ] * index_l + masks + [True, ] * (len(self.pcds) - index_r - 1)
        self.apply_mask(masks)

    def icp_person_per_frame(self, frame, center_pos, mocap_point_cloud):
        radius = 1.
        pc = self.get_cur_pcd(frame, raw_index=True)
        mask1 = ((pc - center_pos) ** 2).sum(axis=1) < radius ** 2
        pc = pc[mask1]

        t = transformation.icp_without_rotation(pc.astype(np.float32), mocap_point_cloud.astype(np.float32),
                                                     pcl.PointCloud(mocap_point_cloud.astype(np.float32)).make_kdtree_flann())
        t = -tw

        t_list = [t, ]
        rt_list = np.identity(4)[np.newaxis, ...].repeat(len([mocap_point_cloud, ]), axis=0)
        for rt, trans in zip(rt_list, t_list):
            rt[:3, 3] = trans
        indexes = [self.to_raw_mocap_index(self.raw_pcd_index_to_cur(i)) for i in range(frame, frame+1, 1)]
        self.apply_mocap_transform(indexes, rt_list)


    def icp_person(self, index_l, index_r, trans_move_limit, dist_threshold, mocap_point_clouds, use_cur_trans=False, fix_trans=False, use_mocap_trans=False, fix_pc=False):
        radius = 1.0
        t_list = []
        mask_list = []
        for frame, v in tqdm.tqdm(zip(range(index_l, index_r + 1, 1), mocap_point_clouds), desc='icp_person', ncols=60):
            pc = self.get_cur_pcd(frame, raw_index=True)

            if use_cur_trans or frame == index_l:
                mocap_transform_index = self.to_raw_mocap_index(self.raw_pcd_index_to_cur(frame))
                assert mocap_transform_index in self.mocaps_transform, f'icp_person:找不到帧：{frame}对应的trans'
                t_init = -self.get_mocap_transform(mocap_transform_index)[:3, 3].astype(np.float32)
            elif use_mocap_trans:
                cur_frame = self.raw_pcd_index_to_cur(frame)
                t_offset = self.mocaps.worldpos(self.to_raw_mocap_index(cur_frame))[0] - \
                           self.mocaps.worldpos(self.to_raw_mocap_index(cur_frame - 1))[0]
                t_init = -(t + t_offset)
            #elif frame == index_l:
                #t_init = transformation.icp_without_rotation(pc.astype(np.float32), mocap_point_clouds[0].astype(np.float32), pcl.PointCloud(mocap_point_clouds[0].astype(np.float32)).make_kdtree_flann())
            else:
                t_init = -t

            if fix_pc:
                mask1 = ((pc - (v.mean(axis=0)-t_init)) ** 2).sum(axis=1) < radius ** 2
                pc = pc[mask1]

            t, m = transformation.mqh_sa_icp_wo_r(v, pc, t_init, trans_move_limit=trans_move_limit, T=1, Tmin=0.1, dist_threshold=dist_threshold,
                                                  k=5, z=t_init[2])  # 611107开头效果很好
            t = -t
            # t = transformation.icp_without_rotation(v, pcds[frame], pcl.PointCloud(pcds[frame].astype(np.float32)).make_kdtree_flann())

            t_list.append(t)

            if fix_pc:
                mask2 = ((pc - (v.mean(axis=0)+t)) ** 2).sum(axis=1) < radius ** 2
                mask_list.append(self.mask_seq_and(np.logical_not(mask1), np.logical_not(mask2)))
                self.person_label_list[frame] = m[mask2]
            else:
                self.person_label_list[frame] = m

        if fix_trans:
            rt_list = np.identity(4)[np.newaxis, ...].repeat(len(mocap_point_clouds), axis=0)
            for rt, trans in zip(rt_list, t_list):
                rt[:3, 3] = trans
            indexes = [self.to_raw_mocap_index(self.raw_pcd_index_to_cur(i)) for i in range(index_l, index_r+1, 1)]
            self.apply_mocap_transform(indexes, rt_list)

        if fix_pc:
            mask_list = [True, ] * index_l + mask_list + [True, ] * (len(self.pcds) - index_r - 1)
            self.apply_mask(mask_list)

    def icp_person_finetune(self, index_l, index_r, trans_move_limit, dist_threshold, mocap_point_clouds, fix_trans=True):
        pcds = [self.get_cur_pcd(frame, raw_index=True) for frame in range(index_l, index_r + 1, 1)]
        init_trans = [-self.get_mocap_transform(self.to_raw_mocap_index(self.raw_pcd_index_to_cur(frame)))[:3, 3].astype(np.float32) \
                      for frame in range(index_l, index_r + 1, 1)]
        z_init = transformation.icp_without_rotation(pcds[0].astype(np.float32),
                                                     mocap_point_clouds[0].astype(np.float32),
                                                     pcl.PointCloud(mocap_point_clouds[0].astype(np.float32)).make_kdtree_flann())[2]

        n = len(pcds)
        m_list, t_list = multiprocess.multi_func(mqh_sa_icp_wo_r, 32, len(pcds), 'icp_person_finetune:', True, pcds, mocap_point_clouds, init_trans, [trans_move_limit, ] * n, [dist_threshold, ] * n, [z_init, ] * n)


        for frame, m in zip(range(index_l, index_r + 1, 1), m_list):
            self.person_label_list[frame] = m

        if fix_trans:
            rt_list = np.identity(4)[np.newaxis, ...].repeat(len(mocap_point_clouds), axis=0)
            for rt, trans in zip(rt_list, t_list):
                rt[:3, 3] = trans
            indexes = [self.to_raw_mocap_index(self.raw_pcd_index_to_cur(i)) for i in range(index_l, index_r+1, 1)]
            self.apply_mocap_transform(indexes, rt_list)

    def icp_object(self, index_l, index_r, trans_limit=1, rotvec_limit=np.pi / 4, ):
        # import socketio
        # client = socketio.Client()
        # client.connect('http://172.18.69.38:5666')

        import time

        threshold = 0.02
        # obj = pipeline_util.read_obj('/SAMSUMG8T/mqh/lidarcapv2/raw/object/雨伞.obj')
        source = pipeline_util.pcd_from_np(self.obj)
        source = source.voxel_down_sample(voxel_size=0.02)

        init_rt = self.object_rt_list[index_l]

        def icp(target, rt, max_iter):
            return o3d.pipelines.registration.registration_icp(
                source, target, threshold, rt,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

        for frame in tqdm.tqdm(range(index_l, index_r + 1, 1), desc='icp_object', ncols=60):
            pc = self.get_cur_pcd(frame, raw_index=True)
            pc = pc[np.logical_not(self.person_label_list[frame])]
            target = pipeline_util.pcd_from_np(pc)
            target.estimate_normals()
            target = target.voxel_down_sample(voxel_size=0.02)

            # init_rt = pipeline_util.ransac_registration(obj, pc, max_iterations=2000,distance_multiplier=1.5, voxel_size=0.04).transformation

            # client.emit('add_pc', ('pc', np.asarray(target.points).tolist()))
            pose, pose_rt = self.get_cur_mocap(self.raw_pcd_index_to_cur(frame))
            # client.emit('add_smpl_mesh', ('human_mesh', pose, None, np.array(pose_rt)[:3, 3].tolist()))

            k = 1

            init_rotvec = R.from_matrix(init_rt[:3, :3]).as_rotvec()
            rt = init_rt.copy()
            rt[:3, 3] = 0
            rt[:3, 3] = target.get_center() - affine(source.get_center()[np.newaxis, ...], rt)[0]
            # print('--------------------------------')
            best_o, best_rt = 0, init_rt

            test = []
            for i in range(k):
                # print('start a new rt:')
                # #client.emit('add_obj', ('obj', init_rt.tolist(), 0))
                o = 0
                reject_count = 3
                for j in range(1000):
                    result = icp(target, rt, 1)
                    new_rt = result.transformation
                    new_rotvec = R.from_matrix(new_rt[:3, :3]).as_rotvec()

                    new_o = result.fitness
                    test.append((new_o, new_rt.copy()))

                    abs_rotvec = np.abs((init_rotvec + np.pi) % np.pi - (new_rotvec + np.pi) % np.pi)
                    rotvec_offset = np.stack((abs_rotvec, np.pi - abs_rotvec)).min(axis=0)

                    if rotvec_offset.max() > rotvec_limit:
                        if reject_count > 0:
                            reject_count -= 1
                        else:
                            test.append((new_rt.copy(), new_rotvec.copy(), rotvec_offset.copy()))
                            # client.emit('add_obj', ('obj', new_rt.tolist(), 0))
                            # print(f'reject by rotvec_limit, {rotvec_offset}, {rotvec_offset.max()} > {rotvec_limit}')
                            break
                    if np.linalg.norm(init_rt[:3, 3] - new_rt[:3, 3]) > trans_limit:
                        if reject_count > 0:
                            reject_count -= 1
                        else:
                            # print(f'reject by trans_limit, {np.linalg.norm(init_rt[:3, 3] - new_rt[:3, 3])} > {trans_limit}')
                            break

                    if new_o > o or reject_count > 0:
                        if new_o <= o:
                            reject_count -= 1
                        # print(f'{o:.3f}->{new_o:.3f} accept')
                        o, rt = new_o, new_rt
                        #client.emit('add_obj', ('obj', rt.tolist(), 0))
                        if o > best_o:
                            best_o, best_rt = o, rt.copy()
                    else:
                        # print(f'{o:.3f}->{new_o:.3f} reject')
                        pass
                        break

                new_rotvec = init_rotvec + (np.random.rand(3) - 0.5) * 2 * rotvec_limit
                new_rt = np.identity(4)
                new_rt[:3, :3] = R.from_rotvec(new_rotvec).as_matrix()
                # new_rt[:3, 3] = init_rt[:3, 3] + (np.random.rand(3) - 0.5) * 2 * 0.1
                new_rt[:3, 3] = target.get_center() - affine(source.get_center()[np.newaxis, ...], new_rt)[0]
                rt = new_rt

                # client.emit('add_obj', ('obj', new_rt.tolist(), 0))
                # print(f'{init_rotvec}->{new_rotvec}')

            # print(f'best o:{best_o}')
            # client.emit('add_obj', ('obj', best_rt.tolist(), 0))
            init_rt = best_rt.copy()
            self.object_rt_list[frame] = best_rt.copy()

    def icp_object_per_frame(self, frame):

        pc = self.get_cur_pcd(frame, raw_index=True)
        assert isinstance(self.person_label_list[frame], np.ndarray), f'找不到帧:{frame}对应的person label!'
        pc = pc[np.logical_not(self.person_label_list[frame])]

        init_rt = pipeline_util.ransac_registration(self.obj, pc, max_iterations=20000, distance_multiplier=1.5, voxel_size=0.04).transformation

        self.object_rt_list[frame] = init_rt

    def get_obj_rt(self, frame):
        return self.object_rt_list[frame]

    def apply_trace_and_cluster_mask(self, index_l, index_r, start_pos, radius, eps=0.2):
        #pcds = self.pcds[start_frame:end_frame]
        masks = []
        center_pos = np.array(start_pos)
        for frame in tqdm.tqdm(range(index_l, index_r+1, 1), desc='Cut Out Person', ncols=60):
            p = self.get_cur_pcd(frame, raw_index=True)
            mask1 = ((p - center_pos) ** 2).sum(axis=1) < radius ** 2
            p = p[mask1]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False))
            #label_with_max_points = np.bincount(labels + 1).argmax() - 1
            #assert label_with_max_points != -1, '在聚类后具有最多点的标签是noice!考虑提高cluster_dbscan的eps值'
            #if label_with_max_points == -1:
            #    print('在聚类后具有最多点的标签是noice!考虑提高cluster_dbscan的eps值')
            #    break
            labels += 1
            if len(np.unique(labels)) == 1:
                human_label = labels[0]
                if human_label == 0:
                    print('在聚类后具有最多点的标签是noice!考虑提高cluster_dbscan的eps值', frame)
                    #break
            else:
                candidate2, candidate1 = np.bincount(labels).argsort()[-2:]
                if candidate1 == 0:
                    human_label = 0
                    print('在聚类后具有最多点的标签是noice!考虑提高cluster_dbscan的eps值', frame)
                    #break
                elif candidate2 == 0:
                    human_label = candidate1
                else:
                    c1_center_pos = p[labels == candidate1].mean(axis=0)
                    c2_center_pos = p[labels == candidate2].mean(axis=0)
                    dist = np.linalg.norm(c1_center_pos[:2] - c2_center_pos[:2])
                    #print('dist', frame, dist)
                    if dist < 0.5:
                        human_label = (candidate1, candidate2)
                    else:
                        # 在最多的两个点标签中选择z轴最大值更大的那个
                        human_label = candidate1 if p[labels == candidate1].max(axis=0)[2] > p[labels == candidate2].max(axis=0)[2] else candidate2

                        # 在最多的两个点标签中选择xy值最接近center_pos的值
                        #human_label = candidate1 if np.linalg.norm((candidate1-center_pos)[:2]) < np.linalg.norm((candidate2-center_pos)[:2]) else candidate2

            #centers = np.vstack([p[labels == i].mean(axis=0) for i in range(-1, labels.max() + 1)])
            #label_with_min_dist = ((centers - center_pos) ** 2).sum(axis=1).argmin()
            #center_pos = centers[label_with_min_dist]

            #mask2 = labels == label_with_max_points
            mask2 = labels == human_label if not isinstance(human_label, tuple) else np.logical_or(labels==human_label[0], labels==human_label[1])
            center_pos = p[mask2].mean(axis=0, keepdims=True)
            #persons.append(p)
            masks.append(self.mask_seq_and(np.logical_not(mask1), np.logical_not(mask2)))

        masks = [True, ] * index_l + masks + [True, ] * (len(self.pcds) - index_r - 1)
        self.apply_mask(masks)

    def apply_quick_trace_and_cluster_mask(self, index_l, index_r, start_pos, radius, eps=0.2):
        #pcds = self.pcds[start_frame:end_frame]
        masks = []
        center_pos = np.array(start_pos)
        for frame in tqdm.tqdm(range(index_l, index_r+1, 1), desc='Cut Out Person', ncols=60):
            p = self.get_cur_pcd(frame, raw_index=True)
            mask1 = ((p - center_pos) ** 2).sum(axis=1) < radius ** 2
            p = p[mask1]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False))
            label_with_max_points = np.bincount(labels + 1).argmax() - 1
            assert label_with_max_points != -1, '在聚类后具有最多点的标签是noice!考虑提高cluster_dbscan的eps值'
            if label_with_max_points == -1:
                print(f'{frame}:{np.unique(labels)}, 在聚类后具有最多点的标签是noice!')
                break

            mask2 = labels != -1
            center_pos = p[mask2].mean(axis=0, keepdims=True)
            #persons.append(p)
            masks.append(self.mask_seq_and(np.logical_not(mask1), np.logical_not(mask2)))

        masks = [True, ] * index_l + masks + [True, ] * (len(self.pcds) - index_r - 1)
        self.apply_mask(masks)

    def apply_cluster_mask(self, index, start_pos, radius, eps=0.2):
        #pcds = self.pcds[start_frame:end_frame]
        masks_r = []
        center_pos = np.array(start_pos)

        index_r = index_l = index

        for frame in tqdm.tqdm(range(index, len(self.pcds), 1), desc='cluster mask', ncols=60):
            p = self.get_cur_pcd(frame, raw_index=True)
            mask1 = ((p - center_pos) ** 2).sum(axis=1) < radius ** 2
            p = p[mask1]
            if len(p) == 0:
                break

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=1, print_progress=False))

            centers = np.vstack([p[labels == i].mean(axis=0) for i in range(labels.min(), labels.max() + 1)])
            label_with_min_dist = ((centers - center_pos) ** 2).sum(axis=1).argmin()
            center_pos = centers[label_with_min_dist]
            mask2 = labels == labels.min() + label_with_min_dist

            m = self.mask_seq_and(np.logical_not(mask1), np.logical_not(mask2))
            masks_r.append(np.logical_not(m))

            index_r = frame

        masks_l = []
        for frame in tqdm.tqdm(range(index-1, 0, -1), desc='cluster mask', ncols=60):
            p = self.get_cur_pcd(frame, raw_index=True)
            mask1 = ((p - center_pos) ** 2).sum(axis=1) < radius ** 2
            p = p[mask1]
            if len(p) == 0:
                break

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=1, print_progress=False))

            centers = np.vstack([p[labels == i].mean(axis=0) for i in range(labels.min(), labels.max() + 1)])
            label_with_min_dist = ((centers - center_pos) ** 2).sum(axis=1).argmin()
            center_pos = centers[label_with_min_dist]
            mask2 = labels == labels.min() + label_with_min_dist

            m = self.mask_seq_and(np.logical_not(mask1), np.logical_not(mask2))
            masks_l.append(np.logical_not(m))

            index_l = frame

        masks = masks_l[::-1] + masks_r

        masks = [True, ] * index_l + masks + [True, ] * (len(self.pcds) - index_r - 1)
        self.apply_mask(masks)


    def cancel_mask(self):
        if len(self.pcds_mask_list) >= 2:
            self.pcds_mask_list.pop()
            return True
        else:
            return False

    def apply_mask(self, mask):
        assert len(mask) == len(self.pcds), f'Try to apply mask(len:{len(mask)}), which is not equal to pcds(len:{len(self.pcds)}'
        self.pcds_mask_list.append(mask)

    def mask_seq_and(self, mask1, mask2):
        if mask1 is True:
            return mask2.copy()
        elif mask2 is True:
            return mask1.copy()
        elif len(mask1) == len(mask2):
            return np.logical_or(mask1, mask2)
        else:
            ids = np.arange(len(mask1))[np.logical_not(mask1)][np.logical_not(mask2)]
            mask = np.ones((len(mask1),), dtype='bool')  # 用bool运行更快
            mask[ids] = False
            return mask

    def get_cur_mask(self, raw_index):
        return reduce(self.mask_seq_and, [e[raw_index] for e in self.pcds_mask_list])


    def get_cur_pcds_map(self):
        cur_map = self.pcds_map_list[0]
        for m in self.pcds_map_list[1:]:
            cur_map = cur_map[m]
        return cur_map

    def get_cur_mocap_map(self):
        cur_map = self.mocaps_map_list[0]
        for m in self.mocaps_map_list[1:]:
            cur_map = cur_map[m]
        return cur_map

    def to_raw_pcd_index(self, index):
        if hasattr(index, "__iter__"):
            index = np.array(index)
            return self.get_cur_pcds_map()[index].tolist()
        else:
            return int(self.get_cur_pcds_map()[index]) if index != -1 else len(self.pcds)

    def raw_pcd_index_to_cur(self, raw_index):
        cur_index = np.where(self.get_cur_pcds_map() == raw_index)
        assert len(cur_index) == 1, '未知错误！'
        return cur_index[0][0]

    def to_raw_mocap_index(self, index):
        if hasattr(index, "__iter__"):
            index = np.array(index)
            return self.get_cur_mocap_map()[index].tolist()
        else:
            return int(self.get_cur_mocap_map()[index]) if index != -1 else len(self.mocaps)

    def apply_pcds_map(self, map):
        assert len(self.pcds_map_list) == 0, '只允许至多一个 pcd map 存在！'
        self.pcds_map_list.append(np.array(map))

    def apply_mocaps_map(self, map):
        assert len(self.mocaps_map_list) == 0, '只允许至多一个 mocap map 存在！'
        self.mocaps_map_list.append(np.array(map))

    def cancel_pcds_map(self):
        if len(self.pcds_map_list) >= 1:
            self.pcds_map_list.pop()
            return True
        else:
            return False

    def cancel_mocaps_map(self):
        if len(self.mocaps_map_list) >= 1:
            self.mocaps_map_list.pop()
            return True
        else:
            return False

    def get_cur_pcd(self, index, raw_index=False):  #raw_index为true表示传入的是raw_index
        if not raw_index:
            index = self.to_raw_pcd_index(index)
        pcd = self.pcds[index]
        pcd = affine(pcd, self.get_cur_transform())
        pcd = pcd[np.logical_not(self.get_cur_mask(index))]
        return pcd

    def get_cur_pcd_and_label(self, index, raw_index=False):  #raw_index为true表示传入的是raw_index
        pcd = self.get_cur_pcd(index, raw_index)
        label = self.person_label_list[index] if raw_index else self.person_label_list[self.to_raw_pcd_index(index)]
        return pcd, label

    def get_cur_pcd_list(self):
        #return multiprocess.multi_func(self.get_cur_pcd, 32, self.get_pcds_len(), 'get cur pcd list:', True, list(range(self.get_pcds_len())))
        return [self.get_cur_pcd(e) for e in tqdm.tqdm(list(range(self.get_pcds_len())), desc='get cur pcd list:')]

    def get_cur_pcd_with_no_rt(self, index, raw_index=False):  #raw_index为true表示传入的是raw_index
        if not raw_index:
            index = self.to_raw_pcd_index(index)
        pcd = self.pcds[index]
        pcd = pcd[np.logical_not(self.get_cur_mask(index))]
        return pcd

    def get_pcds_len(self):
        return len(self.pcds_map_list[-1])

    def get_cur_mocap(self, index, raw_index=False):
        if not raw_index:
            index = self.to_raw_mocap_index(index)

        if index in self.mocap_pose_cache:
            pose = self.mocap_pose_cache[index]
        else:
            pose = self.mocaps.pose(index)
            self.mocap_pose_cache[index] = pose

        return pose.tolist(), self.get_mocap_transform(index).tolist()

    def get_cur_mocap_list(self):
        mocap_map = self.get_cur_mocap_map()
        #poses = multiprocess.multi_func(self.mocaps.pose, 32, len(mocap_map), 'get cur mocap list:', True, mocap_map)
        return [np.array(self.get_cur_mocap(i, raw_index=True)[0]) for i in tqdm.tqdm(mocap_map, desc='get cur mocap list')]


    def get_mocaps_len(self):
        return len(self.mocaps_map_list[-1])

    def access(self, func, args, kwargs):
        return eval(f'self.{func}(*args, **kwargs)')


if __name__ == "__main__":
    from io3d import mocap
    mocaps = mocap.MoCapData('/SAMSUMG8T/mqh/lidarcapv2/lidarcap/mocaps/51805/yxl_chair_05_chr00_rotations.csv', '/SAMSUMG8T/mqh/lidarcapv2/lidarcap/mocaps/51805/yxl_chair_05_chr00_worldpos.csv')
    from io3d.pcd import read_point_cloud
    from util import multiprocess
    filenames = path_util.get_sorted_filenames_by_index('/SAMSUMG8T/mqh/lidarcapv2/lidarcap/pointclouds/51805')
    point_clouds = multiprocess.multi_func(
        read_point_cloud, 32, len(filenames), 'read raw point clouds:', True, filenames)

    seq = seq_process(point_clouds, mocaps)
    seq.get_cur_pcd_list()
    input()

"""
radius = 1
center_pos = np.array(start_pos)
masks = []

center_pos_list = []
pc_list = []

for frame in tqdm.tqdm(range(index_l, index_r+1, 1), desc='Cut Out Person:'):
    p = s.seq.get_cur_pcd(frame, raw_index=True)
    mask1 = ((p - center_pos) ** 2).sum(axis=1) < (radius * 2) ** 2
    p = p[mask1]
    assert mask1.max(), f'cut out person, {frame}frame has no person cloud!'
    t_mocap_pc = transformation.mocap_to_lidar(mocap_point_clouds[frame-index_l], s.seq.get_cur_transform(), p)
    
    center_pos = t_mocap_pc.mean(axis=0, keepdims=True)
    mask2 = ((p - center_pos) ** 2).sum(axis=1) < radius ** 2
    assert mask2.max(), f'cut out person, {frame}frame has no person cloud after icp!'

    masks.append(s.seq.mask_seq_and(np.logical_not(mask1), np.logical_not(mask2)))
    center_pos_list.append(center_pos)
    pc_list.append(p)

#masks = [True, ] * index_l + masks + [True, ] * (len(s.seq.pcds) - index_r - 1)
#s.seq.apply_mask(masks)

def icp(a, b, threshold, trans_init):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(a)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(b)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    return reg_p2p.transformation
frame += 1
pose = s.seq.get_cur_mocap(frame)[0]
v = mocap_point_clouds[frame]
pc = s.seq.get_cur_pcd(frame)
rt = icp(pc, v, 2, np.identity(4))

client.emit('add_pc', ('pc', affine(pc, rt).tolist()))
#client.emit('add_pc', ('pc', pc.tolist()))
client.emit('add_smpl_mesh', ('human_mesh', pose, None, None))
#client.emit('add_pc', ('human_mesh', v.tolist()))


pcds = [s.seq.get_cur_pcd(i, raw_index=True) for i in tqdm.tqdm(range(index_l, index_r+1, 1), desc='get cur pcds')]
pcds_with_no_rt = [s.seq.get_cur_pcd_with_no_rt(i, raw_index=True) for i in tqdm.tqdm(range(index_l, index_r+1, 1), desc='get cur pcds')]

frame += 1
frame = frame % (index_r - index_l + 1)
v, pose = mocap_point_clouds[frame], poses[frame]
t = transformation.get_mocap_to_lidar_translation(v, pcds_with_no_rt[frame], s.seq.get_cur_transform())
client.emit('add_pc', ('pc', pcds[frame].tolist()))
client.emit('add_smpl_mesh', ('human_mesh', pose.tolist(), None, t.tolist()))


import pcl
#frame = 2300
frame += 1
frame = frame % (index_r - index_l + 1)
v, pose = mocap_point_clouds[frame].astype(np.float32), poses[frame]
pc = pcds[frame].astype(np.float32)
t = transformation.icp_without_rotation(v, pcds[frame], pcl.PointCloud(pcds[frame].astype(np.float32)).make_kdtree_flann())
#t = -transformation.icp_without_rotation(pc, v, pcl.PointCloud(v.astype(np.float32)).make_kdtree_flann())
#t = -transformation.mqh_icp_wo_r(v, pc, v.mean(axis=0) - pc.mean(axis=0), trans_move_limit=0.5, T=1, Tmin=0.1, dist_threshold=0.01, k=5, z=-0.86)
#t = -transformation.mqh_icp_wo_r(v, pc, -t, trans_move_limit=0.2, T=1, Tmin=0.5, dist_threshold=0.01, k=5, z=-1)
client.emit('add_pc', ('pc', pc.tolist()))
client.emit('add_smpl_mesh', ('human_mesh', pose.tolist(), None, t.tolist()))

def read_obj(path):
    with open(path) as file:
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

obj = read_obj('/SAMSUMG8T/mqh/lidarcapv2/raw/object/雨伞.obj')

frame += 10
v = mocap_point_clouds[frame]
pc = pcds[frame]
result = ransac_registration(obj, pcds[frame],max_iterations=20000, distance_multiplier=1)
client.emit('add_pc', ('obj', affine(obj, result.transformation).tolist()))


client.emit('add_pc', ('pc', pc.tolist()))
client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, None))

#result = ransac_registration(pc, v,max_iterations=20000)
#client.emit('add_pc', ('pc', affine(pcds[frame], result.transformation).tolist()))
#client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, None))

#result = ransac_registration(v, pc,max_iterations=20000)
#client.emit('add_pc', ('pc', affine(pcds[frame], np.linalg.inv(result.transformation)).tolist()))
#client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, t.tolist()))


"""

"""
import pcl
#frame = 0
if frame == 0:
    v = mocap_point_clouds[frame]
    pc = pcds[frame]
    t = -transformation.icp_without_rotation(pc.astype(np.float32), v.astype(np.float32), pcl.PointCloud(v.astype(np.float32)).make_kdtree_flann())
    
frame += 1
v = mocap_point_clouds[frame]
pc = pcds[frame]

#t_offset = s.seq.mocaps.worldpos(s.seq.to_raw_mocap_index(frame))[0] - s.seq.mocaps.worldpos(s.seq.to_raw_mocap_index(frame-1))[0]
#t += t_offset
print(t)
t = -transformation.mqh_sa_icp_wo_r(v, pc, -t, trans_move_limit=0.2, T=1, Tmin=0.1, dist_threshold=0.01, k=5, z=-1) #611107开头效果很好
#t = -transformation.mqh_icp_wo_r(v, pc, -t, trans_move_limit=0.3, T=1, Tmin=0.1, dist_threshold=0.01, k=5, z=-1)   #611107开头效果很差，跟不上
#t = transformation.icp_without_rotation(v, pcds[frame], pcl.PointCloud(pcds[frame].astype(np.float32)).make_kdtree_flann())

client.emit('add_pc', ('pc', pc.tolist()))
client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, t.tolist()))
"""
"""
poses = s.seq.get_cur_mocap_list()
pcds = s.seq.get_cur_pcd_list()
mocap_point_clouds = s.get_smpl_vertices(poses)

import socketio
client = socketio.Client()
client.connect('http://172.18.69.38:5666')
obj = pipeline_util.read_obj('/SAMSUMG8T/mqh/lidarcapv2/raw/object/雨伞.obj')
a = o3d.geometry.PointCloud()
a.points = o3d.utility.Vector3dVector(obj)
obj_info = pipeline_util.preprocess_point_cloud(a, 0.05)
import pcl
frame = 0
v = mocap_point_clouds[frame]
pc = pcds[frame]
t = -transformation.icp_without_rotation(pc.astype(np.float32), v.astype(np.float32),
                                         pcl.PointCloud(v.astype(np.float32)).make_kdtree_flann())
t_list = []
m_list = []
obj_rt_list = []
import time
for frame in tqdm.tqdm(range(0, 100)):
    v = mocap_point_clouds[frame]
    pc = pcds[frame]

    t, m = transformation.mqh_sa_icp_wo_r(v, pc, -t, trans_move_limit=0.2, T=1, Tmin=0.1, dist_threshold=0.01, k=5, z=-1) #611107开头效果很好
    t = -t
    #t = transformation.icp_without_rotation(v, pcds[frame], pcl.PointCloud(pcds[frame].astype(np.float32)).make_kdtree_flann())
    
    if m.sum() > 0:
        result = pipeline_util.ransac_registration(obj_info, pc[np.logical_not(m)], max_iterations=20000,
                                                   distance_multiplier=1.5, voxel_size=0.04)
        # result = pipeline_util.point_to_point_icp(obj_info[0], pc[~m], 0.1, result.transformation)
        if result.fitness > 0.01:
            client.emit('add_obj', ('obj', result.transformation.tolist(), 0))
            #client.emit('add_pc', ('obj', affine(obj, result.transformation).tolist()))
            obj_rt_list.append(result.transformation)
        else:
            obj_rt_list.append(np.identity(4))
    m_list.append(m)
    t_list.append(t)
    client.emit('add_pc', ('pc', pc.tolist()))
    client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, t.tolist()))
    time.sleep(0.1)
    
    
i = 0
for rt, pc, pose, t in tqdm.tqdm(zip(obj_rt_list, pcds, poses, t_list)):
    client.emit('add_obj', ('obj', rt.tolist(), 0))
    client.emit('add_pc', ('pc', pc.tolist()))
    client.emit('add_smpl_mesh', ('human_mesh', pose.tolist(), None, t.tolist()))
    time.sleep(0.1)
    i += 1
"""
"""
icp_iteration = 100
frame = 65
threshold = 0.02
source = pcd_from_np(obj)
pc = pcds[frame][np.logical_not(m_list[frame])]
target = pcd_from_np(pc)
target.estimate_normals()
source = source.voxel_down_sample(voxel_size=0.02)
target = target.voxel_down_sample(voxel_size=0.02)
#trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7], [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
#source.transform(trans)
result = pipeline_util.ransac_registration(obj, pc, max_iterations=20000,
                                                   distance_multiplier=1.5, voxel_size=0.04)
#cur_obj_rt = np.identity(4)
#cur_obj_rt[:3,3] = target.get_center() - source.get_center()
cur_obj_rt = result.transformation
source.transform(cur_obj_rt)
client.emit('add_pc', ('pc', np.asarray(target.points).tolist()))
client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, t_list[frame].tolist()))
client.emit('add_obj', ('obj', cur_obj_rt.tolist(), 0))
for i in range(icp_iteration):
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
    #print('reg', reg_p2l.transformation)
    source.transform(reg_p2l.transformation)
    cur_obj_rt = reg_p2l.transformation @ cur_obj_rt
    if i % 3 == 0:
        client.emit('add_obj', ('obj', cur_obj_rt.tolist(), 0))
        time.sleep(0.1)
print(i, reg_p2l.fitness)

"""
"""
r = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi / 2, 0, 0])
rt = np.identity(4)
rt[:3, :3] = r
#client.emit('add_obj', ('obj', np.identity(4).tolist(), 0))
client.emit('add_obj', ('obj', rt.tolist(), 0))

"""

"""
#本来是打算搞个模拟退火的物体icp，效果不咋好

icp_iteration = 100
frame = 65
threshold = 0.02
source = pipeline_util.pcd_from_np(obj)
pc = pcds[frame][np.logical_not(m_list[frame])]
target = pipeline_util.pcd_from_np(pc)
target.estimate_normals()
source = source.voxel_down_sample(voxel_size=0.02)
target = target.voxel_down_sample(voxel_size=0.02)
#init_rt = pipeline_util.ransac_registration(obj, pc, max_iterations=2000,distance_multiplier=1.5, voxel_size=0.04).transformation
client.emit('add_pc', ('pc', np.asarray(target.points).tolist()))
client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, t_list[frame].tolist()))
client.emit('add_obj', ('obj', init_rt.tolist(), 0))
from scipy.spatial.transform import Rotation as R
import math
def icp(rt, max_iter):
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, rt,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
T, Tmin, k = 1, 0.01, 100
trans_limit, rotvec_limit = 0.5, np.pi / 2
init_rotvec = R.from_matrix(init_rt[:3, :3]).as_rotvec()
init_rotvec = (init_rotvec + np.pi) % np.pi
rt = init_rt.copy()
o = icp(rt, 1).fitness
print('--------------------------------', o)
#client.emit('add_obj', ('obj', icp(rt, 100).transformation.tolist(), 0))
while T >= Tmin:
    for i in range(k):
        new_rotvec = R.from_matrix(rt[:3, :3]).as_rotvec() + (np.random.rand(3) - 0.5) * 2 * rotvec_limit * T
        new_rotvec = (new_rotvec + np.pi) % np.pi

        new_rt = np.identity(4)
        new_rt[:3, :3] = R.from_rotvec(new_rotvec).as_matrix()
        new_rt[:3, 3] = init_rt[:3, 3] + (np.random.rand(3) - 0.5) * 2 * trans_limit * T
        #rt = new_rt
        rt = icp(new_rt, 1).transformation
        
        rotvec_offset = np.stack((np.abs(init_rotvec - new_rotvec), np.pi - np.abs(init_rotvec - new_rotvec))).min(axis=0)
        if rotvec_offset.max() < rotvec_limit and np.linalg.norm(init_rt[:3, 3] - new_rt[:3, 3]) < trans_limit:
            new_o = icp(rt, 1).fitness
            if new_o - o > 0:
                print(f'{o:.3f}->{new_o:.3f} accept')
                rt, o = new_rt, new_o
            else:
                p = math.exp(((new_o - o) * 10) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    print(f'{o:.3f}->{new_o:.3f} accept by p:{p:.3f}')
                    rt, o = new_rt, new_o
                else:
                    #print(f'{o:.3f}->{new_o:.3f} reject by p:{p:.3f}')
                    pass
                    
        else:
            pass
            #print(f'reject:{rotvec_offset.max()},{init_rotvec}, {new_rotvec} {rotvec_limit}, {np.linalg.norm(init_rt[:3, 3] - new_rt[:3, 3]) > trans_limit}')
    
    T *= 0.6  # 降温函数，也可使用T=0.9
    print('T:', i, T, o)
    client.emit('add_obj', ('obj', rt.tolist(), 0))
client.emit('add_obj', ('obj', rt.tolist(), 0))

"""

"""
import time

frame += 1
threshold = 0.02
source = pipeline_util.pcd_from_np(obj)
pc = pcds[frame][np.logical_not(m_list[frame])]
target = pipeline_util.pcd_from_np(pc)
target.estimate_normals()
source = source.voxel_down_sample(voxel_size=0.02)
target = target.voxel_down_sample(voxel_size=0.02)

#init_rt = pipeline_util.ransac_registration(obj, pc, max_iterations=2000,distance_multiplier=1.5, voxel_size=0.04).transformation

client.emit('add_pc', ('pc', np.asarray(target.points).tolist()))
client.emit('add_smpl_mesh', ('human_mesh', poses[frame].tolist(), None, t_list[frame].tolist()))


from scipy.spatial.transform import Rotation as R
import math


def icp(rt, max_iter):
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, rt,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))


k = 1
trans_limit, rotvec_limit = 1, np.pi / 4
init_rotvec = R.from_matrix(init_rt[:3, :3]).as_rotvec()
rt = init_rt.copy()
rt[:3, 3] = 0
rt[:3, 3] = target.get_center() - affine(source.get_center()[np.newaxis, ...], rt)[0]
print('--------------------------------')
best_o, best_rt = 0, init_rt

test = []
for i in range(k):
    print('start a new rt:')
    #client.emit('add_obj', ('obj', init_rt.tolist(), 0))
    o = 0
    reject_count = 3
    for j in range(1000):
        result = icp(rt, 1)
        new_rt = result.transformation
        new_rotvec = R.from_matrix(new_rt[:3, :3]).as_rotvec()

        new_o = result.fitness
        test.append((new_o, new_rt.copy()))
        
        
        abs_rotvec = np.abs((init_rotvec + np.pi) % np.pi - (new_rotvec + np.pi) % np.pi)
        rotvec_offset = np.stack((abs_rotvec, np.pi - abs_rotvec)).min(axis=0)

        if rotvec_offset.max() > rotvec_limit:
            if reject_count > 0:
                reject_count -= 1
            else:
                test.append((new_rt.copy(), new_rotvec.copy(), rotvec_offset.copy()))
                client.emit('add_obj', ('obj', new_rt.tolist(), 0))
                print(f'reject by rotvec_limit, {rotvec_offset}, {rotvec_offset.max()} > {rotvec_limit}')
                break
        if np.linalg.norm(init_rt[:3, 3] - new_rt[:3, 3]) > trans_limit:
            if reject_count > 0:
                reject_count -= 1
            else:
                print(f'reject by trans_limit, {np.linalg.norm(init_rt[:3, 3] - new_rt[:3, 3])} > {trans_limit}')
                break

        if new_o > o or reject_count > 0:
            if new_o <= o:
                reject_count -= 1
            print(f'{o:.3f}->{new_o:.3f} accept')
            o, rt = new_o, new_rt
            client.emit('add_obj', ('obj', rt.tolist(), 0))
            if o > best_o:
                best_o, best_rt = o, rt.copy()
        else:
            # print(f'{o:.3f}->{new_o:.3f} reject')
            pass
            break

    new_rotvec = init_rotvec + (np.random.rand(3) - 0.5) * 2 * rotvec_limit
    new_rt = np.identity(4)
    new_rt[:3, :3] = R.from_rotvec(new_rotvec).as_matrix()
    # new_rt[:3, 3] = init_rt[:3, 3] + (np.random.rand(3) - 0.5) * 2 * 0.1
    new_rt[:3, 3] = target.get_center() - affine(source.get_center()[np.newaxis, ...], new_rt)[0]
    rt = new_rt
    
    #client.emit('add_obj', ('obj', new_rt.tolist(), 0))
    #print(f'{init_rotvec}->{new_rotvec}')

print(f'best o:{best_o}')
client.emit('add_obj', ('obj', best_rt.tolist(), 0))
init_rt = best_rt.copy()"""