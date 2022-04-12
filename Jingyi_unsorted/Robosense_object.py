# -*- coding: utf-8 -*-
# Author : Zhang.Jingyi
import os
import cv2
import pcl
import numpy as np
import yaml
'''
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
'''


class robosense_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir):
        '''root_dir contains training and testing folders'''

        self.data_dir = os.path.join(root_dir, 'data')

        self.image_dir = os.path.join(self.data_dir, 'image')
        self.lidar_dir = os.path.join(self.data_dir, 'lidar')

    def get_image(self, img_filename):
        return cv2.imread(img_filename)

    def get_lidar(self, lidar_filename):
        # scan = np.fromfile(lidar_filename, dtype=np.float32)
        # scan = scan.reshape((-1,4))
        # edited by xuelun
        scan = pcl.load_XYZI(lidar_filename).to_array()[:, :4]
        return scan

    def get_mocap(self, mocap_info):
        skelenton_3d = np.zeros((69, 3), dtype=float)
        skelenton_3d[0] = np.array(mocap_info[1:4])
        skelenton_3d[1:] = np.array(mocap_info[13:]).reshape((68, 3))
        return skelenton_3d


class Calibration(object):
    ''' Calibration matrices and utils
        create depth image in image2 coord
        Points in <lidar>.pcd are in Robosense coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        Robosense coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z
    '''

    def __init__(self, calib_path):
        # extrinstic_file = os.path.join(calib_path, 'CamExtrinsic.yml')
        # camera_file = os.path.join(calib_path, 'camer.yml')
        # f1 = open(extrinstic_file)
        # f2 = open(camera_file)
        # extrinstic_matrix = yaml.load(f1, Loader=yaml.FullLoader)
        # camera_matrix = yaml.load(f2, Loader=yaml.FullLoader)
        # Projection matrix from rect camera coord to image2 coord
        self.P = np.array([1001.5891335, 0., 953.6128327, 0.,
                           0., 1000.9244526, 582.04816056, 0.,
                           0., 0., 1., 0.])

        # self.P=camera_matrix['intrinsic_matrix']['data']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Robosense coord to reference camera coord
        # edited by xuelun
        self.V2C = np.array(
            [-0.022845605917, -0.99949339352, 0.022159300388, -0.0026512677119,
             -0.048575862249, -0.021029143708, -0.9985980977, 0.24298071602,
             0.99855819254, -0.023889985732, -0.048070829873, 0.16285148859,
             0, 0, 0, 1])
        # self.V2C = extrinstic_matrix['intrinsic_matrix']['data']
        self.V2C = np.reshape(self.V2C, [4, 4])
        self.C2V = inverse_rigid_trans(self.V2C)

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_robo_to_rect(self, pts_3d_robo):
        points = pts_3d_robo[:, :3]
        pts_3d = self.cart2hom(points)  # nx4
        V2C = self.retified_Rotation(self.V2C, -0.04)
        #V2C = self.retified_Translate(V2C, (0, -0.1, 0))
        points_trans = np.dot(pts_3d, np.transpose(V2C))  # nx3
        # edited by xuelun
        # points_trans = np.transpose(np.dot(self.V2C, np.transpose(pts_3d))) # nx3

        final_points = np.hstack(
            (points_trans[:, :3], pts_3d_robo[:, 3:]))  # nx4
        return final_points

    def project_skelenton3d_to_camera(self, skelenton3d):
        skelenton_3d = self.cart2hom(skelenton3d)
        V2C = self.V2C
        skelenton_camera = np.dot(skelenton_3d, np.transpose(V2C))
        return skelenton_camera

    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx4 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        points = pts_3d_rect[:, :3]
        points_3d = self.cart2hom(points)  # nx4
        P = self.retified_Rotation(self.P, 0)
        pts_2d = np.dot(points_3d, np.transpose(P))  # nx3
        pts_2d[:, 0] /= (pts_2d[:, 2] + 1e-8)
        pts_2d[:, 1] /= (pts_2d[:, 2] + 1e-8)
        # edited by xuelun
        # pts_2d = np.transpose(np.dot(self.P, np.transpose(points_3d))) # nx3
        # pts_2d[:,0] /= (pts_2d[:,2]+1e-8)
        # pts_2d[:,1] /= (pts_2d[:,2]+1e-8)
        final_points = np.hstack((pts_2d, pts_3d_rect[:, 3:]))
        return final_points

    def project_skelenton_camera_to_image(self, skelenton_camera):
        skelenton_camera = skelenton_camera[:, :3]
        skelenton_camera_3d = self.cart2hom(skelenton_camera)
        P = self.P
        pts_2d = np.dot(skelenton_camera_3d, np.transpose(P))  # nx3
        pts_2d[:, 0] /= (pts_2d[:, 2] + 1e-8)
        pts_2d[:, 1] /= (pts_2d[:, 2] + 1e-8)
        return pts_2d

    @staticmethod
    def retified_Rotation(M, degree):
        if degree == 0:
            return M
        x0 = x3 = np.cos(degree)
        x1 = -np.sin(degree)
        x2 = -x1
        rotM = np.array([[x0, x1, 0],
                         [x2, x3, 0],
                         [0, 0, 1]])
        sourceM = M[:3, :3]
        targetM = rotM @ sourceM
        M[:3, :3] = targetM
        return M

    @staticmethod
    def retified_Translate(M, meter):
        if meter == 0:
            return M
        x, y, z = meter
        tnsM = np.array([x, y, z])
        sourceM = M[:3, 3]
        targetM = sourceM + tnsM
        M[:3, 3] = targetM
        return M

    # @staticmethod
    # def retified_3D_Rot(M, degree):
    #     if degree == 0: return M
    #     x0 = x3 = np.cos(degree)
    #     x1 = -np.sin(degree)
    #     x2 = -x1
    #     rotM = np.array([[1, 0, 0],
    #                      [0, x0, x1],
    #                      [0, x2, x3]])
    #     sourceM = M[:3,:3]
    #     targetM = rotM @ sourceM
    #     M[:3, :3] = targetM
    #     return M

    def project_robo_to_image(self, pts_3d_robo):
        ''' Input: nx3 points in robosense coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_robo_to_rect(pts_3d_robo)
        pts_2d_imgs = self.project_rect_to_image(pts_3d_rect)
        return pts_2d_imgs

    def project_skelenton3d_to_image(self, skelenton_3d):
        skelenton_camera = self.project_skelenton3d_to_camera(skelenton_3d)
        skelenton_image = self.project_skelenton_camera_to_image(
            skelenton_camera)
        return skelenton_image


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr
