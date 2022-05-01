#   Author: mqh
import os
from util import path_util
import socketio

sio = socketio.AsyncServer(async_mode='tornado')

import open3d as o3d
import asyncio
import argparse
vis = o3d.visualization.Visualizer()
vis.create_window()

@sio.on('*')
def catch_all(event, pid, data):
    print(event, pid, data)

@sio.event
def my_event(sid, data):
    print(sid, data)

@sio.on('my custom event')
def another_event(sid, data):
    print(sid, data)

@sio.event
def connect(sid, environ, auth):
    print('connect ', sid)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)





id_to_geometry = {}

def add_geometry(id, geometry):
    if id in id_to_geometry:
        vis.update_geometry(geometry)
    else:
        vis.add_geometry(geometry)
        id_to_geometry[id] = geometry

@sio.event
def add_pc(sid, id, points, colors=None):
    if id in id_to_geometry:
        pointcloud = id_to_geometry[id]
    else:
        pointcloud = o3d.geometry.PointCloud()

    pointcloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pointcloud.colors = colors

    add_geometry(id, pointcloud)

from smpl import SMPL
import torch
import numpy as np
smpl = SMPL()

from config import SMPL_FILE
import pickle
with open(SMPL_FILE, 'rb') as f:
    smpl_model = pickle.load(f, encoding='iso-8859-1')
    face_index = smpl_model['f'].astype(int)

@sio.event
def add_smpl_pc(sid, id, pose, beta=None, trans=None):
    if beta is None:
        beta = torch.zeros((10,))
    if trans is None:
        trans = torch.zeros((1, 3))
    v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
             torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()
    v += torch.from_numpy(np.array(trans)).float().view(1, 3)
    add_pc(sid, id, v.tolist(), None)

@sio.event
def add_coordinate(sid, id, origin, size):
    axis_pcd = id_to_geometry[id] if id in id_to_geometry else \
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

    add_geometry(id, axis_pcd)


@sio.event
def add_smpl_mesh(sid, id, pose, beta=None, trans=None):
    if beta is None:
        beta = torch.zeros((10,))
    if trans is None:
        trans = torch.zeros((1, 3))
    v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
             torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()
    v += torch.from_numpy(np.array(trans)).float().view(1, 3)

    if id in id_to_geometry:
        m = id_to_geometry[id]
    else:
        m = o3d.geometry.TriangleMesh()


    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(face_index)

    add_geometry(id, m)


async def vis_update():
    while(True):
        vis.poll_events()
        vis.update_renderer()
        await asyncio.sleep(0.01)



add_coordinate(None, 'default_coordinate', [0, 0, 0], 1)

import tornado
app = tornado.web.Application(
    [
        (r"/socket.io/", socketio.get_tornado_handler(sio)),
    ],
    # ... other application options
)
app.listen(5555)
tornado.ioloop.IOLoop.current().run_sync(vis_update)

if __name__ == '__mian__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default=None)
    parser.add_argument('--show_pc', action='store_true',default=False)
    parser.add_argument('--show_smpl_mesh', action='store_true',default=False)
    parser.add_argument('--show_smpl_pc', action='store_true',default=False)
    parser.add_argument('--show_both', action='store_true',default=False)
    args = parser.parse_args()

    root_path = '/SAMSUMG8T/lidarcapv2/lidarcap'
    smpl_path = os.path.join(root_path, 'labels/3d/smpl', args.index)
    pc_path = os.path.join(root_path, 'pointclouds', args.index)
    if args.show_both:
        spml_files = path_util.get_paths_by_suffix(smpl_path, '.json')



"""
test_pose = [0.836127378397543, 0.7968662235677227, 1.3865917901282314, 0.09538421145576546, -0.07802244724603179, -0.0451182395141796, 0.13345783195628988, 0.025858816291387646, -0.012633933396760694, 0.05161666692104735, 0.029963388758322145, -0.055438702473743544, 0.043857358237534706, -0.0003272325108215652, 0.08166379319301671, 0.04381553414104606, 0.07573319563773899, -0.04539729549589903, 0.05115099738541373, 0.03085748656186713, -0.09290075370641884, -0.04113084978589002, 0.15672501019060198, -0.041370933718374754, -0.07885069602835204, -0.06513856421845565, 0.06277148951269874, 0.05217247003534406, 0.1437260905291329, -0.11494412230506264, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007643864863825729, -0.0004676168920657302, 0.12219801993969306, 0.02740611878780802, -0.2250846161100171, -0.3257006962176012, -0.14098278917591137, -0.006157103191784621, 0.056719646616393275, 0.010969569900005296, -0.13450666067288236, 0.07371440274563333, -0.03514749056145958, -0.005092626856970985, -0.9168401000892479, -1.0735153354228397, 0.5103949107809096, -0.4240301374691297, 0.2416073163461026, -0.07199580741350063, -0.08729979701294982, -0.5761975599404016, 0.4247319722219089, -0.12949242787246543, 0.13672831541727248, -0.08995617645605605, 0.11117749205897905, -0.1105943982585262, 0.20086635147617568, 0.02283286859146917, -0.008813476746806785, 0.051946867120518055, -0.3360537860899213, -0.008813358605433249, -0.051946940188505375, 0.33605356229875927]
"""