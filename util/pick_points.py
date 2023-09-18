import open3d as o3d
import numpy as np
import os
import json
import pyransac3d as pyrsc

a = ['Fast', 'Normal', 'Slow']
points_path = '/Users/zhangjingyi/Desktop/tmp/gait/000001.pcd'


save_path = '/Users/zhangjingyi/Desktop/tmp/gait/info_json/93/4/Normal'

if not os.path.exists(save_path):
    os.makedirs(save_path)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def apply_transform(points, RT):
    points = np.asarray(points)
    ones = np.ones((points.shape[0], 1))
    p_w = np.concatenate((points, ones), axis=1)
    p = np.dot(p_w, RT.T)[:, :3]
    print(p.shape)
    return p


def pick_points(pcd, num):
    # selected_points = []
    print("")
    print("1) 请使用至少选择三个对应关系 [shift + 左击]"
    )
    print("   按 [shift + 右击] 撤销拾取的点")
    print("2) 拾取点后，按“Q”关闭窗口")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    # 激活窗口。此函数将阻止当前线程，直到窗口关闭。
    vis.run()  # 等待用户拾取点
    vis.destroy_window()
    print("")
    # picked_point = vis.get_picked_points()[0]
    # selected_points.append(picked_point)
    # assert len(selected_points) == num
    return vis.get_picked_points()


def pick_points_r1(point_cloud):
    # selected_points = []
    # 
    # while len(selected_points) <3:
    #     picked_point = vis.get_picked_points()[0]
    #     selected_points.append(picked_point)
    # 
    # if len(selected_points) == 3:
    #     print(selected_points)
    selected_points = pick_points(point_cloud,3)
    # print(selected_points)
    selected_point_coordinates = [point_cloud.points[i] for i in selected_points]
    ground_normal = np.cross(selected_point_coordinates[1] - selected_point_coordinates[0],
                             selected_point_coordinates[2] - selected_point_coordinates[0])
    ground_normal /= np.sqrt((ground_normal ** 2).sum())

    if cosine_similarity(ground_normal, [0, 0, 1]) < 0:
        ground_normal = -ground_normal
    R = pyrsc.get_rotationMatrix_from_vectors(ground_normal, [0, 0, 1])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -selected_point_coordinates[0].dot(R.T)
    R1 = T
    print('R1 calculated down!')
    save_json = os.path.join(save_path, 'R1.json')
    with open(save_json, 'w') as json_file:
        json.dump(R1.tolist(), json_file)
    
    # point_cloud.points = apply_transform(point_cloud.points, R1)
    # o3d.visualization.draw_geometries([point_cloud])
    point_cloud.transform(R1)
    return point_cloud


def pick_points_r2(point_cloud):
    selected_points = pick_points(point_cloud,2)
    # vis.poll_events()
    # vis.update_renderer()
    # print("请在点云中选择一个点...")
    # vis.poll_events()
    # vis.update_renderer()
    # 
    # if vis.has_gui_events():
    #     gui_event = vis.get_gui_event()
    #     if gui_event and gui_event.get_picked_points():
    # picked_point = vis.get_picked_points()[0]
    # selected_points.append(picked_point)

    if len(selected_points) == 2:
        selected_point_coordinates = [point_cloud.points[i] for i in selected_points]
        axis = selected_point_coordinates[1] - selected_point_coordinates[0]
        axis /= np.sqrt((axis ** 2).sum())

        # if cosine_similarity(axis, [1, 0, 0]) < 0:
        #     axis = -axis
        R = pyrsc.get_rotationMatrix_from_vectors(axis, [1, 0, 0])
        T = np.eye(4)
        T[:3, :3] = R
        R2 = T

        save_json = os.path.join(save_path, 'R2.json')
        with open(save_json, 'w') as json_file:
            json.dump(R2.tolist(), json_file)

        # point_cloud.points = apply_transform(point_cloud.points, R2)
        # o3d.visualization.draw_geometries([point_cloud])
        # vis.update_geometry(point_cloud)
        point_cloud.transform(R2)
        return point_cloud
    
    
def pick_points_5points(point_cloud):
    # global selected_points
    selected_points = pick_points(point_cloud,5)
    # vis.poll_events()
    # vis.update_renderer()
    # print("请在点云中选择一个点...")
    # vis.poll_events()
    # vis.update_renderer()
    # 
    # if vis.has_gui_events():
    #     gui_event = vis.get_gui_event()
    #     if gui_event and gui_event.get_picked_points():
    #         picked_point = gui_event.get_picked_points()[0]
    # picked_point=vis.get_picked_points()[0]
    # selected_points.append(picked_point)

    if len(selected_points) == 5:
        selected_point_coordinates = [point_cloud.points[i] for i in selected_points]
        x_min = selected_point_coordinates[0][0]
        x_max = selected_point_coordinates[1][0]
        y_min = selected_point_coordinates[2][1]
        y_max = selected_point_coordinates[3][1]
        first_position = selected_point_coordinates[4]

        results = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'first_position': first_position.tolist()
        }
        save_json = os.path.join(save_path, 'points.json')
        with open(save_json, 'w') as json_file:
            json.dump(results, json_file)
        # vis.destroy_window()
        
        
# point_cloud = o3d.geometry.PointCloud()
point_cloud = o3d.io.read_point_cloud(points_path)

o3d.visualization.draw_geometries_with_editing([])

point_cloud = pick_points_r1(point_cloud)
point_cloud = pick_points_r2(point_cloud)
pick_points_5points(point_cloud)

# 注册选择点的回调函数
# o3d.visualization.draw_geometries_with_animation_callback([point_cloud], pick_points_r1)
# o3d.visualization.draw_geometries_with_animation_callback([point_cloud], pick_points_r2)
# o3d.visualization.draw_geometries_with_animation_callback([point_cloud], pick_points_5points)

# 关闭窗口
o3d.visualization.draw_geometries(destroy_window=True)
