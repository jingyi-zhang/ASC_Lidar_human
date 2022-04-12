# -*- coding: utf-8 -*-
# Author : Zhang.Jingyi
'''
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
'''
import numpy as np
import Robosense_object as datas
import os
import cv2
import sys
import argparse
import csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

##############################
# data structure:
# ROOT_PATH contains data dir:
#   Image
#   IMU
#   Lidar
##############################
parser = argparse.ArgumentParser()
parser.add_argument('--ROOT_PATH', type=str, default='/mnt/d/human_data/0724/frame_data',
                    help='the dir contains data')
parser.add_argument('--mode', type=str, default='depth_image', help='choose from:'
                    '[depth_image,mocap_image]')
args = parser.parse_args()


def get_lidar_in_image_fov(pc_robo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=0.0):
    ''' 
    Filter lidar points, keep those in image FOV 
    clip_distance is the closest distance
    '''
    pts_2d = calib.project_robo_to_image(pc_robo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_robo[:, 0] > clip_distance)
    imgfov_pc_robo = pc_robo[fov_inds, :]
    if return_more:
        return imgfov_pc_robo, pts_2d, fov_inds
    else:
        return imgfov_pc_robo


def show_lidar_on_image(pc_robo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pts_3d, pts_2d, fov_inds = get_lidar_in_image_fov(pc_robo,
                                                             calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pts_rect = calib.project_robo_to_rect(imgfov_pts_3d)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    #########################################################
    # 可视化投影图
    # for i in range(imgfov_pts_2d.shape[0]):
    #     # depth = imgfov_pc_rect[i,2]
    #     # color = cmap[int(640.0/depth),:]
    #     # color = cmap[int(255*depth/max_depth)]
    #     color = [34,34,34]
    #     cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
    #         int(np.round(imgfov_pts_2d[i,1]))),
    #         2, color=tuple(color), thickness=-1)

    depth_img = np.zeros_like(img)
    intensity_img = np.zeros_like(img)
    depth_np = np.zeros_like(img)
    intensity_np = np.zeros_like(img)

    #intensity_indx = (imgfov_pc_rect[:,3]<255)& (imgfov_pc_rect[:,3]>220)
    max_depth = max(imgfov_pts_rect[:, 2])
    max_intensity = max(imgfov_pts_rect[:, 3])

    for i in range(imgfov_pts_2d.shape[0]):
        x = int(np.floor(imgfov_pts_2d[i, 0]))
        y = int(np.floor(imgfov_pts_2d[i, 1]))
        depth = imgfov_pts_rect[i, 2]
        intensity = imgfov_pts_rect[i, 3]

        depth_color = cmap[int(255 * depth / max_depth)]
        intensity_color = cmap[int(255 * intensity / max_intensity)]

        cv2.circle(depth_img, (x, y), 2, color=tuple(
            depth_color), thickness=-1)

        cv2.circle(intensity_img, (x, y), 2, color=tuple(
            intensity_color), thickness=-1)

        depth_np[y, x] = depth
        intensity_np[y, x] = intensity

    return img, depth_img, intensity_img, depth_np, intensity_np


def depth_image(img_filename, lidar_filename, depth_path, depth_numpy):
    # calib = datas.Calibration()
    dataset = datas.robosense_object(ROOT_DIR)
    print(img_filename)
    img = dataset.get_image(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    pc_robo = dataset.get_lidar(lidar_filename)
    img, depth_img, intensity_img, depth_np, intensity_np = show_lidar_on_image(
        pc_robo, img, calib, img_width, img_height)

    cv2.imwrite(depth_path, depth_img)
    #cv2.imwrite(intensity_path, intensity_img)
    np.save(depth_numpy, depth_np)
    #np.save(intensity_numpy, intensity_np)


def show_skelenton_on_image(skelenton, img, calib):
    ''' Project skelenton to image '''
    skelenton_2d = calib.project_skelenton3d_to_image(skelenton)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(skelenton_2d.shape[0]):
        color = [168, 168, 168]
        cv2.circle(img, (int(np.round(skelenton_2d[i, 0])),
                         int(np.round(skelenton_2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)
    return img


def mocap_image(image_file, mocap_info, calib, save_path):
    dataset = datas.robosense_object(ROOT_DIR)
    img = dataset.get_image(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    skeleton_3d = dataset.get_mocap(mocap_info)
    img = show_skelenton_on_image(skeleton_3d, img, calib)
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    ROOT_PATH = args.ROOT_PATH
    img_path = os.path.join(ROOT_PATH, 'images')
    lidar_path = os.path.join(ROOT_PATH, 'lidar')
    depth_path = os.path.join(ROOT_PATH, 'depth')
    intensity_path = os.path.join(ROOT_PATH, 'intensity')
    intensity_numpy_path = os.path.join(ROOT_PATH, 'intensity-numpy')
    depth_numpy_path = os.path.join(ROOT_PATH, 'depth-numpy')
    mocap_path = os.path.join(ROOT_PATH, 'mocap')
    calib_path = os.path.join(ROOT_PATH, 'calib')
    skelenton_2d = os.path.join(ROOT_PATH, 'skelenton_output')
    calib = datas.Calibration(calib_path)

    from tqdm import tqdm
    if args.mode == 'depth_image':
        # for _, _ , files in os.walk(img_path):
        #     for file in tqdm(files):
        #         indx = file.strip('.jpg').strip('frame')
        #         lidar_index = str(int(indx)//6+373)
        #         image_file = os.path.join(img_path, file)
        #         lidar_file = os.path.join(lidar_path, lidar_index+'.pcd')
        #         depth_file = os.path.join(depth_path, file)
        #         intensity_file = os.path.join(intensity_path, file)
        #         depth_numpy = os.path.join(depth_numpy_path, indx)
        #         intensity_numpy = os.path.join(intensity_numpy_path, indx)
        #         depth_image(image_file, lidar_file, depth_file, intensity_file,
        #                     depth_numpy, intensity_numpy)
        image_file = os.path.join('data/image/00859.jpg')
        lidar_file = os.path.join(
            'data/2021-07-24-16-02-45-RS-0-Data-10000-12000/536.pcd')
        depth_file = 'c529a186613fe044c67998e8ed0c7b2.jpg'
        depth_numpy = 'c529a186613fe044c67998e8ed0c7b2'
        depth_image(image_file, lidar_file, depth_file, depth_numpy)

    # if args.mode == 'mocap_image':
    #     mocap_csv = os.path.join(mocap_path, 'worldpos.csv')
    #     with open(mocap_csv, 'r') as f:
    #         reader = csv.reader(f)
    #         mocap_info = list(reader)
    #     for _,_, files in os.walk(img_path):
    #         numbers = sorted([ int(x[5:-4]) for x in files])[:-25]
    #         for image_index in tqdm(numbers):
    #             mocap_index = int(image_index)-28 #defined based on datasets
    #             image_file = os.path.join(img_path, f'frame{image_index}.jpg')
    #             skelenton_2d_file = os.path.join(skelenton_2d, f'frame{image_index}.jpg')
    #             mocap_image(image_file, mocap_info[mocap_index], calib, skelenton_2d_file)
