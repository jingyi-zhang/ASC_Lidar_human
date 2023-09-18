# -*- coding: utf-8 -*-
# @Author  : jingyi

import socketio
import h5py
import time
import os
import glob
import open3d as o3d
import numpy as np
from tqdm import tqdm

# for i in range(18, 105):
i = 24
for j in [1, 2, 3, 4]:
    for y in ['Fast', 'Normal', 'Slow']:
        idx = str(i) + '/' + str(j) + '/' + y
        print(idx)
        input_pcds_path = os.path.join('/hdd24T/zjy/jingyi/gait/xmu_gait/lidarcap/human', idx)
        filenames = glob.glob(os.path.join(input_pcds_path, '*.pcd'))
        filenames.sort(key=lambda e: int(os.path.basename(e).split('.')[0]))
        human = []
        for file in filenames:
            pcd = o3d.io.read_point_cloud(file)
            human.append(np.array(pcd.points))
            
        client = socketio.Client()
        client.connect('http://127.0.0.1:8785')
        time.sleep(3)
        
        for i_ in tqdm(range(len(human))):
            # j, k = i // 16, i % 16
            # print(i)
            client.emit('add_pc',
                        ('pc1', human[i_].tolist(), [255 / 255.0, 128 / 255.0, 0 / 255.0]))
            # client.emit('add_pc', (
            # 'pc2', background[i].tolist(), [255 / 255.0, 227 / 255.0, 0 / 255.0]))
            # client.emit('add_pc',
            #             ('pc3', shadow[i].tolist(), [83 / 255.0, 245 / 255.0, 185 / 255.0]))
            time.sleep(0.1)
        client.disconnect()

