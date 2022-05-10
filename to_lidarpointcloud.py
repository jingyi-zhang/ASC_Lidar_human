import functools
import os
import open3d as o3d
import numpy as np
from glob import glob
import sys
import shutil
from util import multiprocess


def hidden_point_removal(pcd, camera=[0, 0, 0]):
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    # camera = [view_point[0], view_point[0], diameter]
    # camera = view_point
    dist = np.linalg.norm(pcd.get_center())
    # radius = diameter * 100
    radius = dist * 150

    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)
    return pcd


def select_points_on_the_scan_line(points, view_point=None, scans=126, line_num=1801,
                                   fov_up=12.5, fov_down=-12.5, precision=1.1):
    fov_up = np.deg2rad(fov_up)
    fov_down = np.deg2rad(fov_down)
    fov = abs(fov_down) + abs(fov_up)

    # ratio = fov/(scans - 1)   # 64bins 的竖直分辨率
    # hoz_ratio = 2 * np.pi / (line_num - 1)    # 64bins 的水平分辨率
    # precision * np.random.randn()
    ratio = np.deg2rad(0.2)
    hoz_ratio = np.deg2rad(0.2)

    print(points.shape[0])
    if view_point is not None:
        points -= view_point
    depth = np.linalg.norm(points, 2, axis=1)
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # pc_ds = []

    saved_box = {s: {} for s in np.arange(scans)}

    #### 筛选fov范围内的点
    for idx in range(0, points.shape[0]):
        rule1 = pitch[idx] >= fov_down
        rule2 = pitch[idx] <= fov_up
        rule3 = abs(pitch[idx] % ratio) < ratio * 0.4
        rule4 = abs(yaw[idx] % hoz_ratio) < hoz_ratio * 0.4
        if rule1 and rule2:
            scanid = np.rint((pitch[idx] + 1e-4) / ratio) + scans // 2
            pointid = np.rint((yaw[idx] + 1e-4) // hoz_ratio)

            if pointid > 0 and scan_x[idx] < 0:
                pointid += line_num // 2
            elif pointid < 0 and scan_y[idx] < 0:
                pointid += line_num // 2

            z = np.sin(scanid * ratio + fov_down)
            xy = abs(np.cos(scanid * ratio + fov_down))
            y = xy * np.sin(pointid * hoz_ratio)
            x = xy * np.cos(pointid * hoz_ratio)

            # 找到根指定激光射线夹角最小的点
            cos_delta_theta = np.dot(points[idx], np.array([x, y, z])) / depth[idx]
            delta_theta = np.arccos(abs(cos_delta_theta))
            if pointid in saved_box[scanid]:
                if delta_theta < saved_box[scanid][pointid]['delta_theta']:
                    saved_box[scanid][pointid].update(
                        {'points': points[idx], 'delta_theta': delta_theta})
            else:
                saved_box[scanid][pointid] = {'points': points[idx],
                                              'delta_theta': delta_theta}

    save_points = []
    for key, value in saved_box.items():
        if len(value) > 0:
            for k, v in value.items():
                save_points.append(v['points'])

                # pc_ds = np.array(pc_ds)
    save_points = np.array(save_points)

    #####
    print(save_points.shape)
    pc = o3d.open3d.geometry.PointCloud()
    pc.points = o3d.open3d.utility.Vector3dVector(save_points)
    pc.paint_uniform_color([0.5, 0.5, 0.5])
    pc.estimate_normals()

    return pc

def simulatorLiDAR(root, out_root):
    index = root.split('/')[-1]
    print(index)
    out_pose_path = os.path.join(out_root, 'pose', index)
    out_segment_path = os.path.join(out_root, 'segment', index)
    if not os.path.exists(out_pose_path):
        os.makedirs(out_pose_path)
    if not os.path.exists(out_segment_path):
        os.makedirs(out_segment_path)

    mocap2lidar_matrix = np.array([
        [-1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]])

    behave2lidarcap = np.array([
        [ 0.34114292, -0.81632519,  0.46608443, 13.04092283],
        [-0.93870972, -0.26976026,  0.21460074, -1.25513142],
        [-0.04945297, -0.5107275, -0.85831922,  2.25963742],
        [ 0.,  0.,  0.,  1.]])

    for _, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            dir_path = os.path.join(root, dirname)
            person_path = os.path.join(dir_path, 'person', 'fit02', 'person_fit.ply')
            pose_path = os.path.join(dir_path, 'person', 'fit02', 'person_fit.pkl')
            for _,dirnames_son, _ in os.walk(dir_path):
                assert len(dirnames_son) == 2
                if dirnames_son[0] == 'person':
                    object_name = dirnames_son[1]
                else:
                    object_name = dirnames_son[0]
                break
            object_path = os.path.join(dir_path, object_name, 'fit01', object_name+'_fit.ply')

            print(f'Process {person_path}')
            person_point_clouds = o3d.open3d.io.read_triangle_mesh(person_path)
            object_point_clouds = o3d.open3d.io.read_triangle_mesh(object_path)

            # persaon & object points read
            if len(person_point_clouds.triangles) > 0:
                person_point_clouds.compute_vertex_normals()
                person_point_clouds = person_point_clouds.sample_points_poisson_disk(100000)
            else:
                person_point_clouds = o3d.io.read_point_cloud(person_point_clouds)

            if len(object_point_clouds.triangles) > 0:
                object_point_clouds.compute_vertex_normals()
                object_point_clouds = object_point_clouds.sample_points_poisson_disk(100000)
            else:
                object_point_clouds = o3d.io.read_point_cloud(object_point_clouds)

            # points translation & rotation
            object_point_clouds = object_point_clouds.transform(mocap2lidar_matrix)
            person_point_clouds = person_point_clouds.transform(mocap2lidar_matrix)
            object_point_clouds = object_point_clouds.transform(behave2lidarcap)
            person_point_clouds = person_point_clouds.transform(behave2lidarcap)

            # hidden points removal
            person_point_clouds = hidden_point_removal(person_point_clouds)
            object_point_clouds = hidden_point_removal(object_point_clouds)
            assert np.asarray(person_point_clouds.points).shape[0] > 30
            point_clouds = np.concatenate((np.asarray(person_point_clouds.points), np.asarray(object_point_clouds.points)))

            #downsample
            point_clouds = select_points_on_the_scan_line(point_clouds)

            #save_results path
            save_pose_filename = os.path.join(out_pose_path, dirname.split('.')[0]+'.ply')
            save_segment_filename = os.path.join(out_segment_path, dirname.split('.')[0]+'.pcd')
            save_pose_gt = os.path.join(out_pose_path,dirname.split('.')[0]+'.pkl')

            # o3d.io.write_point_cloud(save_path, pcd)
            o3d.io.write_point_cloud(save_segment_filename, point_clouds)
            shutil.copyfile(person_path, save_pose_filename)
            shutil.copyfile(pose_path, save_pose_gt)
        break

if __name__ == '__main__':
    simulatorLiDAR(
        '/mnt/d/squeeze/sequences/Date01_Sub01_boxlong_hand', '/mnt/d/human/human_downsample/labels/3d')
