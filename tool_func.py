import open3d as o3d
import numpy as np
import cv2
import sys
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import math
import random
import pickle as pkl
import argparse
import json
import paramiko
from subprocess import run

from visualization.Timer import Timer
from o3dvis import o3dvis, Keyword
from pypcd import pypcd


class Segmentation():
    def __init__(self, pcd):
        self.pcd = pcd
        self.segment_models = {}
        self.segments = {}

    def seg_by_normals(self, eps=0.15):
        """
        eps (float):  Density parameter that is used to find neighbouring points.
        """
        normal_segments = {}
        normal_pcd = o3d.geometry.PointCloud()
        normal_pcd.points = self.pcd.normals
        points_number = np.asarray(self.pcd.points).shape[0]
        lables = np.array(normal_pcd.cluster_dbscan(
            eps=0.15, min_points=points_number // 20))
        for i in range(lables.max() + 1):
            colors = plt.get_cmap("tab20")(i)
            normal_segments[i] = self.pcd.select_by_index(
                list(np.where(lables == i)[0]))
            normal_segments[i].paint_uniform_color(list(colors[:3]))
        rest = self.pcd.select_by_index(list(np.where(lables == -1)[0]))
        # rest_idx = list(np.where(lables == -1)[0])
        return normal_segments, rest

    def loop_ransac(self, seg, count=0, max_plane_idx=5, distance_threshold=0.02):
        rest = seg
        plane_count = 0
        for i in range(max_plane_idx):
            colors = plt.get_cmap("tab20")(i + count)
            points_number = np.asarray(rest.points).shape[0]
            if points_number < 50:
                break

            self.segment_models[i + count], inliers = rest.segment_plane(
                distance_threshold=distance_threshold, ransac_n=3, num_iterations=1200)

            if len(inliers) < 50:
                break

            self.segments[i + count] = rest.select_by_index(inliers)
            self.segments[i + count].paint_uniform_color(list(colors[:3]))
            plane_count += 1
            rest = rest.select_by_index(inliers, invert=True)
            # print("pass", i, "/", max_plane_idx, "done.")
        return rest, plane_count

    def dbscan_with_ransac(self, seg, count, max_plane_idx=5, distance_threshold=0.02):
        rest = seg
        rest_idx = []
        d_threshold = distance_threshold * 10
        plane_count = 0

        for i in range(max_plane_idx):
            colors = plt.get_cmap("tab20")(count + i)

            points_number = np.asarray(rest.points).shape[0]
            if points_number < 50:
                break

            equation, inliers = rest.segment_plane(
                distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
            inlier_pts = rest.select_by_index(inliers)

            if len(inliers) < 50:
                break

            self.segment_models[i + count] = equation
            self.segments[i + count] = inlier_pts

            labels = np.array(
                self.segments[i + count].cluster_dbscan(eps=d_threshold, min_points=15))
            candidates = [len(np.where(labels == j)[0])
                          for j in np.unique(labels)]
            if len(labels) <= 0:
                print('======================')
                print('======Catch u=========')
                print('======================')
            best_candidate = int(np.unique(labels)[np.where(
                candidates == np.max(candidates))[0][0]])
            # print("the best candidate is: ", best_candidate)
            rest = rest.select_by_index(
                inliers, invert=True) + self.segments[i + count].select_by_index(
                list(np.where(labels != best_candidate)[0]))
            # rest_idx = inliers[np.where(labels != best_candidate)[0]]
            self.segments[i + count] = self.segments[i + count].select_by_index(
                list(np.where(labels == best_candidate)[0]))
            self.segments[i + count].paint_uniform_color(list(colors[:3]))
            plane_count += 1

            # print("pass", i+1, "/", max_plane_idx, "done.")
        return rest, plane_count

    def filter_plane(self):
        equations = []
        planes = []
        rest = []
        for i in range(len(self.segments)):
            pts_normals = np.asarray(self.segments[i].normals)
            points_number = pts_normals.shape[0]
            error_degrees = abs(
                np.dot(pts_normals, self.segment_models[i][:3]))
            # error_degrees < cos20°(0.93969)
            if np.sum(
                    error_degrees < 0.9063) > points_number * 0.25 or points_number < 200:
                print('Max error degrer：', math.acos(
                    min(error_degrees)) * 180 / math.pi)
                print(
                    f'More than {(np.sum(error_degrees < 0.93969) / (points_number / 100)):.2f}%, sum points {points_number}')
                rest.append(self.segments[i])
            else:
                equations.append(self.segment_models[i])
                planes.append(self.segments[i])
        return equations, planes, rest

    def dbscan(self, rest):
        labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=10))
        max_label = labels.max()
        for i in range(max_label):
            list[np.where(labels == i)[0]]
        print(f"point cloud has {max_label + 1} clusters")

        colors = plt.get_cmap("tab10")(
            labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

    def run(self, max_plane=5, distance_thresh=0.02):
        normal_segments, rest = self.seg_by_normals()
        count = 0
        rest_pts = [rest]
        for i in range(len(normal_segments)):
            # rest, plane_count = self.loop_ransac(normal_segments[i], count, max_plane, distance_thresh)
            rest, plane_count = self.dbscan_with_ransac(
                normal_segments[i], count, max_plane, distance_thresh)
            count += plane_count
            rest_pts.append(rest)
        # self.dbscan(rest)
        equations, planes, rest_plane = self.filter_plane()
        return equations, planes, rest_pts + rest_plane


# if __name__ == "__main__":
#     Vector3dVector = o3d.utility.Vector3dVector
#     import pyransac3d as pyrsc
#     pcd_file = "E:\\SCSC_DATA\HumanMotion\\1023\\normal_seg_test.pcd"
#     pcd = o3d.io.read_point_cloud(pcd_file)
#     scene_pcd = pcd.voxel_down_sample(voxel_size=0.02)
#     scene_points = np.asarray(scene_pcd.points)
#     scene_colors = np.asarray(scene_pcd.colors)
#     scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=100))
#     scene_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     # ==============================================
#     # Ransac 平面拟合
#     # ==============================================
#     seg_pcd = Segmentation(scene_pcd)
#     plane_equations, segments, rest = seg_pcd.run(8, 0.03)

#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(window_name='RT', width=1280, height=720)
#     print('Sum plane: ', len(segments))
#     for i in range(len(segments)):
#         vis.add_geometry(segments[i])
#     # for i, g in enumerate(rest):
#     #     # if i == 1:
#     #     vis.add_geometry(g)

#     while True:
#             # o3dcallback()
#         vis.poll_events()
#         vis.update_renderer()
#         cv2.waitKey(10)


def cal_checkpoiont():
    # start point
    start = np.array([-0.012415, -0.021619, 0.912602])  # mocap
    b = np.array([-0.014763, -0.021510, 0.910694])  # mocap + lidar + filt
    c = np.array([-0.006008, -0.020295, 0.923018])  # slamcap

    d2 = np.linalg.norm(b - start)
    d3 = np.linalg.norm(c - start)
    print('start point in labbuilding')
    print('d2 ', d2)
    print('d3 ', d3)

    # end point. 12224
    a = np.array([-1.820513, -2.804140, 0.912960])  # mocap
    b = np.array([0.162808, -0.190505, 0.889144])  # mocap + lidar
    bb = np.array([0.139104, -0.109615, 0.861548])  # frame:12224  + filt
    c = np.array([0.147751, -0.105651, 0.921784])

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d3 = np.linalg.norm(c - start)
    print('end point in labbuilding')

    print('d1 ', d1)
    print('d2 ', d2)
    print('d3 ', d3)

    # frame 1612
    a = np.array([0.028645, -0.679929, 0.912945])  # mocap
    b = np.array([-0.024122, 0.028871, 0.975404])  # mocap + lidar
    bb = np.array([0.003263, -0.047339, 0.939227])  # filt
    c = np.array([0.040516, -0.068466, 0.917972])  # slamcap

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d22 = np.linalg.norm(bb - start)
    d3 = np.linalg.norm(c - start)
    print('CP1 point in labbuilding')

    print('d1 ', d1)
    print('d2 ', d2)
    print('d22 ', d22)
    print('d3 ', d3)

    # 9367
    start = np.array([57.381191, -25.599457, 1.035680])  # ground
    a = np.array([5.5611e+01, -2.4989e+01, -1.5531e-02])  # mocap
    b = np.array([57.2693, -25.4669, 0.9961])  # mocap+ ldiar
    bb = np.array([57.3397, -25.5093, 0.9585])  # mocap+ ldiar +filt
    c = np.array([57.3179, -25.5093, 1.0339])
    print('cp2 point in labbuilding')

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d22 = np.linalg.norm(bb - start)
    d3 = np.linalg.norm(c - start)
    print('d1 ', d1)
    print('d2 ', d2)
    print('d22 ', d22)
    print('d3 ', d3)

    # ============= campus ===========
    # start
    start = np.array([0.006690, 0.016684, 0.913541])  # mocap
    b = np.array([-0.030292, -0.130614, 0.940346])  # mocap + lidar
    bb = np.array([0.005844, 0.015523, 0.899268])  # mocap + lidar +filt
    c = np.array([0.003294, -0.059318, 0.937367])  # SLAMCap

    # 4599
    a = np.array([-1.819105, 0.616108, 0.913044])  # mocap
    b = np.array([0.068938, 0.905881, 0.925295])  # mocap + lidar
    bb = np.array([0.059497, 0.693883, 0.905785])  # mocap + lidar +filt
    c = np.array([0.029597, 0.912938, 0.930939])  # SLAMCap

    d1 = np.linalg.norm(a - start)
    d2 = np.linalg.norm(b - start)
    d22 = np.linalg.norm(bb - start)
    d3 = np.linalg.norm(c - start)
    print('end point in campus')

    print('d1 ', d1)
    print('d2 ', d2)
    print('d22 ', d22)
    print('d3 ', d3)


def cal_dist(file_path, idx):
    traj = np.loadtxt(file_path)
    st = traj[0, 0]
    idx = int(idx - st)
    dist = np.linalg.norm(traj[1:idx + 1, 1:4] - traj[:idx, 1:4], axis=1).sum()
    print(f'{idx + st}: {dist:.3f}')


# _dir = 'E:\\SCSC_DATA\\HumanMotion\\HPS\\hps_txt'

# tot_dist = []
# txtfiles = os.listdir(_dir)
# count = 0
# for i, txt in enumerate(txtfiles):
#     if txt.split('_')[-1] == 'trans.txt':
#         count +=1
#         traj = np.loadtxt(os.path.join(_dir, txt))
#         dist = np.linalg.norm(traj[1:] - traj[:-1], axis=1).sum()
#         tot_dist.append(dist)
#         print(f'num {count} dist: {dist:.3f}')


def cal_mocap_smpl_trans():
    # slamcap, 1927 frams campus
    a = np.array([32.472198, 60.644207, -0.657228])
    # slamcap , 1832 frams campus
    aa = np.array([34.384434, 53.580666, -0.561240])
    # slamcap , 2949 frams campus
    aaa = np.array([-25.074957, 63.042736, 0.064208])

    # mocap, 1927 frames(10088) campus
    b = np.array([33.435452, 56.857124, 0.926840])
    # mocap, 1832 frames(9609) campus
    bb = np.array([34.662342, 50.486450, 0.900601])
    # mocap, 2949 frames(15162) campus
    bbb = np.array([-22.015713, 62.341846, 0.886846])
    print('1927: ', b - a)
    print('1832: ', bb - aa)
    print('2949: ', bbb - aaa)


# print('meand dist: ', np.asarray(tot_dist).mean())
# cal_checkpoiont()
# cal_mocap_smpl_trans()
# file_path = 'E:\\SCSC_DATA\\HumanMotion\\visualization\\lab_building_lidar_filt_synced_offset.txt'
# cal_dist(file_path, 1612)
# cal_dist(file_path, 12224)
# cal_dist(file_path, 9367)
# cal_dist(file_path, 9265)


def make_bounding_box():
    """
    # 测试open3d的boundingbox函数
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(
        "C:\\Users\\Daiyudi\\Desktop\\temp\\001121.pcd")
    people = o3d.io.read_point_cloud(
        "C:\\Users\\Daiyudi\\Desktop\\temp\\001121.ply")
    print(pcd)  # 输出点云点的个数
    print(people)  # 输出点云点的个数
    aabb = people.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)  # aabb包围盒为红色
    obb = people.get_oriented_bounding_box()
    obb.color = (0, 1, 0)  # obb包围盒为绿色

    seg_pcd = pcd.crop(obb)
    bg = pcd.select_by_index(
        obb.get_point_indices_within_bounding_box(pcd.points), invert=True)
    axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=aabb.get_center())
    axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=obb.get_center())
    axis2 = axis2.rotate(obb.R)
    people.paint_uniform_color([0, 0, 1])
    seg_pcd.paint_uniform_color([1, 0, 1])
    bg.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries(
        [bg, people, seg_pcd, axis1, axis2, aabb, obb])


def read_pcd_from_server(filepath):
    hostname = "10.24.80.240"
    port = 511
    username = 'dyd'
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, compress=True)
    sftp_client = client.open_sftp()
    # "/hdd/dyd/lidarcap/velodyne/crop_6/000001.pcd"
    remote_file = sftp_client.open(filepath, mode='rb')  # 文件路径

    try:
        pc_pcd = pypcd.PointCloud.from_fileobj(remote_file)
        pc = np.zeros((pc_pcd.pc_data.shape[0], 4))
        pc[:, 0] = pc_pcd.pc_data['x']
        pc[:, 1] = pc_pcd.pc_data['y']
        pc[:, 2] = pc_pcd.pc_data['z']
        pc[:, 3] = pc_pcd.pc_data['intensity']
        return pc
    except Exception as e:
        print(f"Load {filepath} error")
    finally:
        remote_file.close()


def imges_to_video(path):
    """输入文件夹图片所在文件夹, 利用ffmpeg生成视频

    Args:
        path (str): [图片所在文件夹]
    """
    strs = path.split('\\')[-1]
    # strs = sys.argv[2]
    video_path = os.path.join(os.path.dirname(path), strs + '.mp4')
    video_path2 = os.path.join(os.path.dirname(path), strs + '.avi')

    # command = f"ffmpeg -f image2 -i {path}\\{strs}_%4d.jpg -b:v 10M -c:v h264 -r 20  {video_path}"
    command = f"ffmpeg -f image2 -i {path}\\%4d.jpg -b:v 10M -c:v h264 -r 30  {video_path2}"
    if not os.path.exists(video_path) and not os.path.exists(video_path2):
        run(command, shell=True)


def read_pkl_file(file_name):
    """
    Reads a pickle file
    Args:
        file_name:
    Returns:
    """
    with open(file_name, 'rb') as f:
        data = pkl.load(f)
    return data


def save_hps_file_to_txt(hps_filepath):
    """ 把hps的文件转成txt, 方便可视化
    Args:
        hps_filepath (str): 包含hps的文件夹
    """
    results_list = os.listdir(hps_filepath)
    for r in results_list:
        rf = os.path.join(hps_filepath, r)
        if r.split('.')[-1] == 'pkl':
            results = read_pkl_file(filepath)
            trans = results['transes']
            np.savetxt(filepath.split('.')[0] + '.txt', trans)
            print('save: ', filepath.split('.')[0] + '.txt')


lidar_cap_view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": 'false',
    "trajectory":
        [
            {
                "boundingbox_max": [14.874032020568848, 76.821174621582031,
                                    3.0000000000000004],
                "boundingbox_min": [-15.89847469329834, -0.17999999999999999,
                                    -9.3068475723266602],
                "field_of_view": 60.0,
                "front": [-0.31752652860060709, -0.88020048903111714,
                          0.3527378668986792],
                "lookat": [-4.6463824978204959, 2.0940846184404625,
                           0.24133203465013156],
                "up": [0.077269992865328138, 0.34673394080716397, 0.9347753326307483],
                "zoom": 0.059999999999999998
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def load_scene(pcd_path, scene_name):
    print('Loading scene...')
    scene_pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, scene_name + '.pcd'))
    # print('Loading normals...')
    # normal_file = os.path.join(pcd_path, scene_name + '_normals.pkl')
    # if os.path.exists(normal_file):
    #     with open(normal_file, 'rb') as f:
    #         normals = pkl.load(f)
    #     scene_pcd.normals = o3d.utility.Vector3dVector(normals)
    # else:
    #     scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=80))
    #     normals = np.asarray(scene_pcd.normals)
    #     with open(normal_file, 'wb') as f:
    #         pkl.dump(normals, f)
    #     print('Save scene normals in: ', normal_file)

    # scene_pcd.voxel_down_sample(voxel_size=0.02)
    print('Scene loaded...')
    return scene_pcd


def visulize_scene_with_meshes(plydir, pcd_dir, scene_name):
    """ 载入场景点云和生成的human meshes

    Args:
        plydir (str): [description]
        pcd_dir (str): [description]
        scene_name (str): [description]
    """
    scene_pcd = load_scene(pcd_dir, scene_name)
    vis = o3dvis()
    vis.add_scene_gemony(scene_pcd)

    mesh_list = []
    sphere_list = []

    while True:
        with Timer('update renderer', True):
            # o3dcallback()
            vis.waitKey(10, helps=False)
            if Keyword.READ:
                # 读取mesh文件
                meshfiles = os.listdir(plydir)
                for mesh in mesh_list:
                    vis.remove_geometry(mesh, reset_bounding_box=False)
                mesh_list.clear()
                mesh_l1 = []  # 无优化的结果
                mesh_l2 = []  # 动捕的原始结果
                mesh_l3 = []  # 优化的结果
                mesh_l4 = []  # 其他类型
                for plyfile in meshfiles:
                    if plyfile.split('.')[-1] != 'ply':
                        continue
                    # print('name', plyfile)
                    if plyfile.split('_')[-1] == 'opt.ply':
                        mesh_l1.append(plyfile)
                    elif plyfile.split('_')[-1] == 'mocap.ply':
                        mesh_l2.append(plyfile)
                    elif plyfile.split('_')[-1] == 'smpl.ply':
                        mesh_l3.append(plyfile)
                    else:
                        mesh_l4.append(plyfile)

                with open(
                        'J:\\Human_motion\\visualization\\info_video_climbing.json') as f:
                    info = json.load(f)['climbing_top_2']

                mesh_list += vis.add_mesh_by_order(plydir, mesh_l1, 'red',
                                                   'wo_opt_' + info['name'],
                                                   start=info['start'], end=info['end'],
                                                   info=info)
                mesh_list += vis.add_mesh_by_order(plydir, mesh_l2, 'yellow',
                                                   'mocap_' + info['name'],
                                                   start=info['start'], end=info['end'],
                                                   info=info)
                mesh_list += vis.add_mesh_by_order(plydir, mesh_l3, 'blue',
                                                   'with_opt_' + info['name'],
                                                   start=info['start'], end=info['end'],
                                                   info=info)
                mesh_list += vis.add_mesh_by_order(plydir, mesh_l4,
                                                   'blue', 'others' + info['name'],
                                                   order=False)

                Keyword.READ = False

            sphere_list = vis.visualize_traj(plydir, sphere_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hps-file', action='store_true',
                        help='run directory', )
    parser.add_argument('--remote-dir', action='store_true',
                        help='The remoted directory of pcd files')
    parser.add_argument('--velodyne-dir', action='store_true',
                        help='The local directory of pcd files')
    parser.add_argument('--img-dir', action='store_true',
                        help='A directroy containing imgs')
    parser.add_argument('--mesh-dir', action='store_true',
                        help='A directroy containing human mesh files')
    parser.add_argument('--scene-dir', action='store_true',
                        help='A directroy containing scene pcd files')
    parser.add_argument('--scene', action='store_true',
                        help="scene's file name")

    # 输入待读取的文件夹
    parser.add_argument('--file-name', '-f', type=str,
                        help='A file or a directory', default=None)
    args, opts = parser.parse_known_args()

    hps_file = 'E:\\SCSC_DATA\\HumanMotion\\HPS\\result'
    velodyne_dir = 'C:\\Users\\Daiyudi\\Desktop\\temp\\semantic_26'
    scene_dir = 'J:\\Human_motion\\visualization'
    mesh_dir = 'J:\\Human_motion\\visualization\\climbinggym002_step_1'
    scene = 'climbinggym002'
    remote_dir = '/hdd/dyd/lidarcap/velodyne/6'

    # 读取hps的轨迹，保存成txt
    if args.hps_file:
        if args.file_name:
            hps_file = args.file_name
        save_hps_file_to_txt(hps_file)

    # 将图片保存成视频
    if args.img_dir:
        imges_to_video(args.file_name)

    # 可视化分割结果的点云
    if args.velodyne_dir:
        # pcd visualization
        if args.file_name:
            velodyne_dir = args.file_name
        vis = o3dvis()
        vis.visulize_point_clouds(velodyne_dir, skip=0, view=lidar_cap_view)

    # 可视化场景和human mesh
    if args.mesh_dir:
        if args.file_name:
            mesh_dir = args.file_name
        visulize_scene_with_meshes(mesh_dir, scene_dir, scene)

    # 可视化远程的文件夹
    if args.remote_dir:
        # visualize the pcd files on the remote server
        if args.file_name:
            remote_dir = args.file_name
        vis = o3dvis()
        vis.visulize_point_clouds(remote_dir, skip=50, view=lidar_cap_view, remote=True)

