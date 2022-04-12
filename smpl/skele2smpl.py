# -*- coding: utf-8 -*-
# @Author  : jingyi
'''
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
'''
import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_PATH)
from io3d import mocap
from smpl import SMPL

import argparse
import generate_ply
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--bvh_name', type=str)
args = parser.parse_args()

'''
pose --> rotation of 24 skelentons
beta --> shape of human

pose can be:
    1. (B, 24, 3, 3)
    or
    2. (B, 72)
beta should be:
    (B, 10)
'''

'''
SMPL
'Root', 'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
'Left_Finger', 'Right_Finger'
'''


if __name__ == '__main__':

    mocap_data = mocap.MoCapData(
        'data/rectangular_worldpos.csv', 'data/rectangular_rotations.csv')

    n = len(mocap_data)
    print(n)
    smpl = SMPL()

    for i in range(726, n):
        vertices = smpl(torch.from_numpy(mocap_data.pose(
            i)).unsqueeze(0).float(), torch.zeros((1, 10)))
        print(vertices.shape)
        generate_ply.save_ply(vertices, 'data/vertices_smpl.ply')
        exit()
