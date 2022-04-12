# -*- coding: utf-8 -*-
# @Author  : jingyi
'''
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
'''
from __future__ import division
from numpy.core.arrayprint import array2string

import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
import config as cfg
try:
    import cPickle as pickle
except ImportError:
    import pickle
from geometric_layers import rodrigues

POSE_DIM = 72
BETA_DIM = 10


class SMPL(nn.Module):

    def __init__(self):
        super(SMPL, self).__init__()
        model_file = cfg.SMPL_FILE
        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding='iso-8859-1')
        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i, v,
                                                                     J_regressor_shape).to_dense())
        self.register_buffer(
            'weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer(
            'posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer(
            'v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs',
                             torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('faces',
                             torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.register_buffer('kintree_table', torch.from_numpy(
            smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in
                     range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in
             range(1, self.kintree_table.shape[1])]))

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        J_regressor_extra = torch.from_numpy(
            np.load(cfg.JOINT_REGRESSOR_TRAIN_EXTRA)).float()
        self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.joints_idx = cfg.JOINTS_IDX

    def forward(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :].to(device)
        shapedirs = self.shapedirs.view(-1, 10)[None,
                                                :].expand(batch_size, -1, -1).to(device)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor.to(device), v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1).to(device)
        posedirs = self.posedirs.view(-1, 207)[None,
                                               :].expand(batch_size, -1, -1).to(device)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890,
                                                                              3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1).to(device)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(
            batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(
            batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights.to(device), G.permute(1, 0, 2, 3).contiguous(
        ).view(24, -1)).view(6890, batch_size, 4, 4).transpose(0, 1)
        rest_shape_h = torch.cat(
            [v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_joints(self, vertices, return_numpy=True):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        if type(vertices) == np.ndarray:
            vertices = torch.from_numpy(vertices)
        joints = torch.einsum(
            'bik,ji->bjk', [vertices, self.J_regressor.to(vertices.device)])
        if return_numpy:
            joints = joints.cpu().numpy()
        return joints
        # joints_extra = torch.einsum(
        #     'bik,ji->bjk', [vertices, self.J_regressor_extra])
        # joints = torch.cat((joints, joints_extra), dim=1)
        # joints = joints[:, cfg.JOINTS_IDX]
        # return joints

    def get_vertices(self, pose=None, beta=None, trans=None, return_numpy=True):

        def get_batched_tensor(array, dim):
            if array is None:
                return torch.zeros((1, dim))
            else:
                if type(array) == np.ndarray:
                    res = torch.from_numpy(array)
                elif type(array) == torch.Tensor:
                    res = array.clone()
                else:
                    sys.stderr.write('type wrong')
                res = res.float()
                if res.ndimension() == 1:
                    res.unsqueeze_(0)
            return res

        pose_tensor = get_batched_tensor(pose, POSE_DIM).float()
        beta_tensor = get_batched_tensor(beta, BETA_DIM).float()
        res = self.forward(pose_tensor, beta_tensor)
        res.squeeze_(0)
        if trans is not None:
            res += torch.from_numpy(trans).to(res.device)
        if return_numpy:
            res = res.cpu().numpy()
        return res
