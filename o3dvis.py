import open3d as o3d
import numpy as np
import cv2
import sys
import os
import paramiko
from pypcd import pypcd


def client_server(username='dyd', hostname="10.24.80.241", port=911):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, compress=True)
    return client


def list_dir_remote(client, folder):
    stdin, stdout, stderr = client.exec_command('ls ' + folder)
    res_list = stdout.readlines()
    return [i.strip() for i in res_list]


def read_pcd_from_server(client, filepath):
    sftp_client = client.open_sftp()
    remote_file = sftp_client.open(filepath, mode='rb')  # 文件路径

    try:
        pc_pcd = pypcd.PointCloud.from_fileobj(remote_file)
        pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
        pc[:, 0] = pc_pcd.pc_data['x']
        pc[:, 1] = pc_pcd.pc_data['y']
        pc[:, 2] = pc_pcd.pc_data['z']
        if pc_pcd.fields[-1] == 'rgb':
            append = pypcd.decode_rgb_from_pcl(pc_pcd.pc_data['rgb']) / 255
        else:
            append = pc_pcd.pc_data[pc_pcd.fields[-1]].reshape(-1, 1)

        return np.concatenate((pc, append), axis=1)
    except Exception as e:
        print(f"Load {filepath} error")
    finally:
        remote_file.close()


colors = {
    'yellow': [251 / 255, 217 / 255, 2 / 255],
    'red': [234 / 255, 101 / 255, 144 / 255],
    'blue': [27 / 255, 158 / 255, 227 / 255],
    'purple': [61 / 255, 79 / 255, 222 / 255],
    'blue2': [75 / 255, 145 / 255, 183 / 255],
}


class Keyword():
    PAUSE = True  # pause the visualization
    DESTROY = False  # destory window
    REMOVE = False  # remove all geometies
    READ = False  # read the ply files
    VIS_TRAJ = False  # visualize the trajectory
    SAVE_IMG = False  # save the images in open3d window
    SET_VIEW = False  # set the view based on the info
    VIS_STREAM = True  # only visualize the the latest mesh stream
    ROTATE = False  # rotate the view automatically


def o3d_callback_rotate():
    Keyword.ROTATE = not Keyword.ROTATE
    return False


camera = {
    'phi': 0,
    'theta': -30,
    'cx': 0.,
    'cy': 0.5,
    'cz': 3.}


def init_camera(camera_pose):
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    # init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    init_param.extrinsic = np.array(camera_pose)
    ctr.convert_from_pinhole_camera_parameters(init_param)


def set_camera(camera_pose):
    theta, phi = np.deg2rad(-(camera['theta'] + 90)), np.deg2rad(camera['phi'] + 180)
    theta = theta + np.pi
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    rot_x = np.array([
        [1., 0., 0.],
        [0., ct, -st],
        [0, st, ct]
    ])
    rot_z = np.array([
        [cp, -sp, 0],
        [sp, cp, 0.],
        [0., 0., 1.]
    ])
    camera_pose[:3, :3] = rot_x @ rot_z
    return camera_pose


def get_camera():
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    return np.array(init_param.extrinsic)


def o3dcallback(camera_pose=None):
    if ROTATE:
        camera['phi'] += np.pi / 10
        camera_pose = set_camera(get_camera())
        # camera_pose = np.array([[-0.927565, 0.36788, 0.065483, -1.18345],
        #                         [0.0171979, 0.217091, -0.976, -0.0448631],
        #                         [-0.373267, -0.904177, -0.207693, 8.36933],
        #                         [0, 0, 0, 1]])
    print(camera_pose)
    init_camera(camera_pose)


def set_view(vis):
    Keyword.SET_VIEW = not Keyword.SET_VIEW
    print('SET_VIEW', Keyword.SET_VIEW)
    return False


def save_imgs(vis):
    Keyword.SAVE_IMG = not Keyword.SAVE_IMG
    print('SAVE_IMG', Keyword.SAVE_IMG)
    return False


def stream_callback(vis):
    # 以视频流方式，更新式显示mesh
    Keyword.VIS_STREAM = not Keyword.VIS_STREAM
    print('VIS_STREAM', Keyword.VIS_STREAM)
    return False


def pause_callback(vis):
    Keyword.PAUSE = not Keyword.PAUSE
    # print('Pause', Keyword.PAUSE)
    return False


def destroy_callback(vis):
    Keyword.DESTROY = not Keyword.DESTROY
    return False


def remove_scene_geometry(vis):
    Keyword.REMOVE = not Keyword.REMOVE
    return False


def read_dir_ply(vis):
    Keyword.READ = not Keyword.READ
    print('READ', Keyword.READ)
    return False


def read_dir_traj(vis):
    Keyword.VIS_TRAJ = not Keyword.VIS_TRAJ
    print('VIS_TRAJ', Keyword.VIS_TRAJ)
    return False


def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False


def load_render_option(vis):
    vis.get_render_option().load_from_json(
        "../../test_data/renderoption.json")
    return False


def capture_depth(vis):
    depth = vis.capture_depth_float_buffer()
    plt.imshow(np.asarray(depth))
    plt.show()
    return False


def capture_image(vis):
    image = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(image))
    plt.show()
    return False


def print_help(is_print=True):
    if is_print:
        print('============Help info============')
        print('Press R to refresh visulization')
        print('Press Q to quit window')
        print('Press D to remove the scene')
        print('Press T to load and show traj file')
        print('Press F to stop current motion')
        print('Press . to turn on auto-screenshot ')
        print('Press , to set view zoom based on json file ')
        print('Press SPACE to pause the stream')
        print('=================================')


class o3dvis():
    def __init__(self, window_name='DAI_VIS'):
        self.init_vis(window_name)
        print_help()

    def add_scene_gemony(self, geometry):
        if not Keyword.REMOVE:
            self.add_geometry(geometry)

    def init_vis(self, window_name):

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord(" "), pause_callback)
        self.vis.register_key_callback(ord("Q"), destroy_callback)
        self.vis.register_key_callback(ord("D"), remove_scene_geometry)
        self.vis.register_key_callback(ord("R"), read_dir_ply)
        self.vis.register_key_callback(ord("T"), read_dir_traj)
        self.vis.register_key_callback(ord("F"), stream_callback)
        self.vis.register_key_callback(ord("."), save_imgs)
        self.vis.register_key_callback(ord(","), set_view)
        self.vis.create_window(window_name=window_name, width=1280, height=720)

    def waitKey(self, key, helps=True):
        print_help(helps)
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            cv2.waitKey(key)
            if Keyword.DESTROY:
                self.vis.destroy_window()
            if not Keyword.PAUSE:
                break
        return Keyword.READ

    def add_geometry(self, geometry, reset_bounding_box=True, waitKey=10):
        self.vis.add_geometry(geometry, reset_bounding_box)
        self.waitKey(waitKey, helps=False)

    def remove_geometry(self, geometry, reset_bounding_box=True):
        self.vis.remove_geometry(geometry, reset_bounding_box)

    def set_view_zoom(self, info, count, steps):
        """根据参数设置vis的视场角

        Args:
            vis ([o3d.visualization.VisualizerWithKeyCallback()]): [description]
            info ([type]): [description]
            count ([int]): [description]
            steps ([int]): [description]
        """
        ctr = self.vis.get_view_control()
        elements = ['zoom', 'lookat', 'up', 'front', 'field_of_view']
        if 'step1' in info.keys():
            steps = info['step1']
        if 'views' in info.keys() and 'steps' in info.keys():
            views = info['views']
            fit_steps = info['steps']
            count += info['start']
            for i, v in enumerate(views):
                if i == len(views) - 1:
                    continue
                if count >= fit_steps[i + 1]:
                    continue
                for e in elements:
                    z1 = np.array(views[i]['trajectory'][0][e])
                    z2 = np.array(views[i + 1]['trajectory'][0][e])
                    if e == 'zoom':
                        ctr.set_zoom(z1 + (count - fit_steps[i]) * (z2 - z1) / (
                                    fit_steps[i + 1] - fit_steps[i]))
                    elif e == 'lookat':
                        ctr.set_lookat(z1 + (count - fit_steps[i]) * (z2 - z1) / (
                                    fit_steps[i + 1] - fit_steps[i]))
                    elif e == 'up':
                        ctr.set_up(z1 + (count - fit_steps[i]) * (z2 - z1) / (
                                    fit_steps[i + 1] - fit_steps[i]))
                    elif e == 'front':
                        ctr.set_front(z1 + (count - fit_steps[i]) * (z2 - z1) / (
                                    fit_steps[i + 1] - fit_steps[i]))
                break

        elif 'trajectory' in info.keys():
            self.vis.reset_view_point(True)
            ctr.set_zoom(np.array(info['trajectory'][0]['zoom']))
            ctr.set_lookat(np.array(info['trajectory'][0]['lookat']))
            ctr.set_up(np.array(info['trajectory'][0]['up']))
            ctr.set_front(np.array(info['trajectory'][0]['front']))

        return False
        # for e in elements:
        #     if count > steps:
        #         break
        #     z1 = np.array(info['view1']['trajectory'][0][e])
        #     z2 = np.array(info['view2']['trajectory'][0][e])
        #     if e == 'zoom':
        #         ctr.set_zoom(z1 + count * (z2-z1) / (steps - 1))
        #     elif e == 'lookat':
        #         ctr.set_lookat(z1 + count * (z2-z1) / (steps - 1))
        #     elif e == 'up':
        #         ctr.set_up(z1 + count * (z2-z1) / (steps - 1))
        #     elif e == 'front':
        #         ctr.set_front(z1 + count * (z2-z1) / (steps - 1))
        #     # elif e == 'field_of_view':
        #     ctr.change_field_of_view(z1 + count * (z2-z1) / (steps - 1))

    def add_mesh_together(self, plydir, mesh_list, color):
        """_summary_

        Args:
            plydir (_type_): _description_
            mesh_list (_type_): _description_
            color (_type_): _description_
        """
        geometies = []
        for mesh_file in mesh_list:
            plyfile = os.path.join(plydir, mesh_file)
            # print(plyfile)
            mesh = o3d.io.read_triangle_mesh(plyfile)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(colors[color])
            # mesh.vertices = Vector3dVector(np.array(mesh.vertices) - trajs[num[i],1:4] + mocap_trajs[num[i],1:4])
            geometies.append(mesh)
            self.add_geometry(mesh, reset_bounding_box=False, waitKey=0)
        return geometies

    def add_mesh_by_order(self, plydir, mesh_list, color, strs='render', order=True,
                          start=None, end=None, info=None):
        """[summary]

        Args:
            plydir ([str]): directory name of the files
            mesh_list ([list]): file name list
            color ([str]): [red, yellow, green, blue]
            strs (str, optional): [description]. Defaults to 'render'.
            order (bool, optional): [description]. Defaults to True.
            start ([int], optional): [description]. Defaults to None.
            end ([int], optional): [description]. Defaults to None.
            info ([type], optional): [description]. Defaults to None.
        Returns:
            [list]: [A list of geometries]
        """
        save_dir = os.path.join(plydir, strs)

        if order:
            num = np.array([int(m.split('_')[0]) for m in mesh_list], dtype=np.int32)
            idxs = np.argsort(num)
        else:
            idxs = np.arange(len(mesh_list))
        pre_mesh = None

        geometies = []
        helps = True
        count = 0

        # trajs = np.loadtxt('G:\\Human_motion\\visualization\\trajs\\campus_lidar_filt_synced_offset.txt')
        # mocap_trajs = np.loadtxt('G:\\Human_motion\\visualization\\trajs\\mocap_trans_synced.txt')

        sphere_list = []
        for i in idxs:
            # set view zoom
            if info is not None and Keyword.SET_VIEW:
                self.set_view_zoom(info, count, end - start)
            if order and end > start:
                if num[i] < start or num[i] > end:
                    continue

            plyfile = os.path.join(plydir, mesh_list[i])
            # print(plyfile)
            mesh = o3d.io.read_triangle_mesh(plyfile)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(colors[color])
            # mesh.vertices = Vector3dVector(np.array(mesh.vertices) - trajs[num[i],1:4] + mocap_trajs[num[i],1:4])
            if Keyword.VIS_STREAM and pre_mesh is not None:
                self.remove_geometry(pre_mesh, reset_bounding_box=False)
                geometies.pop()
            Keyword.VIS_STREAM = True  #
            geometies.append(mesh)
            self.add_geometry(mesh, reset_bounding_box=False)

            # if count % 5 == 0:
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            #     sphere.vertices = Vector3dVector(np.asarray(sphere.vertices) + trajs[num[i],1:4])
            #     sphere.compute_vertex_normals()
            #     sphere.paint_uniform_color(color)
            #     sphere_list.append(sphere)
            #     self.add_geometry(sphere, reset_bounding_box = False)

            pre_mesh = mesh
            if not self.waitKey(10, helps=helps):
                break
            helps = False
            self.save_imgs(save_dir, strs + '_{:04d}.jpg'.format(count))
            count += 1

        for s in sphere_list:
            self.remove_geometry(s, reset_bounding_box=False)

        return geometies

    def visualize_traj(self, plydir, sphere_list):
        """[读取轨迹文件]

        Args:
            plydir ([str]): [description]
            sphere_list ([list]): [description]
        """
        if not Keyword.VIS_TRAJ:
            return sphere_list

        for sphere in sphere_list:
            self.remove_geometry(sphere, reset_bounding_box=False)
        sphere_list.clear()
        traj_files = os.listdir(plydir)

        # 读取文件夹内所有的轨迹文件
        for trajfile in traj_files:
            if trajfile.split('.')[-1] != 'txt':
                continue
            print('name', trajfile)
            if trajfile.split('_')[-1] == 'offset.txt':
                color = 'red'
            elif trajfile.split('_')[-1] == 'synced.txt':
                color = 'yellow'
            else:
                color = 'blue'
            trajfile = os.path.join(plydir, trajfile)
            trajs = np.loadtxt(trajfile)[:, 1:4]
            traj_cloud = o3d.geometry.PointCloud()
            # show as points
            traj_cloud.points = Vector3dVector(trajs)
            traj_cloud.paint_uniform_color(color)
            sphere_list.append(traj_cloud)
            # for t in range(1400, 2100, 1):
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            #     sphere.vertices = Vector3dVector(np.asarray(sphere.vertices) + trajs[t])
            #     sphere.compute_vertex_normals()
            #     sphere.paint_uniform_color(color)
            #     sphere_list.append(sphere)

        # 轨迹可视化
        for sphere in sphere_list:
            self.add_geometry(sphere, reset_bounding_box=False)

        Keyword.VIS_TRAJ = False
        return sphere_list

    def set_view(self, view):
        ctr = self.vis.get_view_control()
        if view is not None:
            # self.vis.reset_view_point(True)
            ctr.set_zoom(np.array(view['trajectory'][0]['zoom']))
            ctr.set_lookat(np.array(view['trajectory'][0]['lookat']))
            ctr.set_up(np.array(view['trajectory'][0]['up']))
            ctr.set_front(np.array(view['trajectory'][0]['front']))
            return True
        return False

    def save_imgs(self, out_dir, filename):
        """[summary]

        Args:
            out_dir ([str]): [description]
            filename ([str]): [description]
        """
        outname = os.path.join(out_dir, filename)
        if Keyword.SAVE_IMG:
            os.makedirs(out_dir, exist_ok=True)
            self.vis.capture_screen_image(outname)

    def visulize_point_clouds(self, file_path, skip=150, view=None, remote=False):
        """visulize the point clouds stream

        Args:
            file_path (str): [description]
            skip (int, optional): Defaults to 150.
            view (dict): A open3d format viewpoint,
                         you can get one view by using 'ctrl+c' in the visulization window.
                         Default None.
        """
        if remote:
            client = client_server()
            files = sorted(list_dir_remote(client, file_path))
        else:
            files = sorted(os.listdir(file_path))

        pointcloud = o3d.geometry.PointCloud()
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3,
                                                                     origin=[0, 0, 0])

        self.add_geometry(axis_pcd)
        self.add_geometry(pointcloud)

        Reset = True

        mesh_list = []

        for i, file_name in enumerate(files):
            if i < skip:
                continue

            for mesh in mesh_list:
                self.remove_geometry(mesh, reset_bounding_box=False)
            mesh_list.clear()

            if file_name.endswith('.txt'):
                pts = np.loadtxt(os.path.join(file_path, file_name))
                pointcloud.points = o3d.utility.Vector3dVector(pts[:, :3])
            elif file_name.endswith('.pcd') or file_name.endswith('.ply'):
                if remote:
                    pcd = read_pcd_from_server(client, file_path + '/' + file_name)
                    pointcloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
                    if pcd.shape[1] == 6:
                        pointcloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
                else:
                    pcd = o3d.io.read_point_cloud(os.path.join(file_path, file_name))
                    pointcloud.points = pcd.points
                    pointcloud.colors = pcd.colors

                    # ! Temp code, for visualization test
                    mesh_dir = os.path.join(os.path.join(os.path.dirname(
                        file_path), 'instance_human'), file_name.split('.')[0])
                    if os.path.exists(mesh_dir):
                        mesh_list += self.add_mesh_together(
                            mesh_dir, os.listdir(mesh_dir), 'blue')

            else:
                continue

            self.vis.update_geometry(pointcloud)

            if Reset:
                Reset = self.set_view_zoom(view, 0, 0)

            self.waitKey(10, helps=False)
            self.save_imgs(os.path.join(file_path, 'imgs'),
                           '{:04d}.jpg'.format(i - skip))
        if remote:
            client.close()
        while True:
            self.waitKey(10, helps=False)

