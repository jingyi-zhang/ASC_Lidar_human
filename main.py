from ast import parse
from matplotlib.pyplot import axis
from util import transformation
from io3d import mocap, obj, pcd
from tqdm import tqdm
import argparse
import cv2
import math
import numpy as np
import os
import shutil


def dai():
    # Dai
    intrinsic_matrix = np.array(
        [1515.72, 0, 1103.06,
         0, 1513.97, 632.042,
         0, 0, 1]
    ).reshape(3, 3)

    extrinsic_matrix = np.array([
        -0.927565, 0.36788, 0.065483, -1.18345,
        0.0171979, 0.217091, -0.976, -0.0448631,
        -0.373267, -0.904177, -0.207693, 8.36933,
        0, 0, 0, 1
    ]).reshape(4, 4)

    distortion_coefficients = np.array(
        [0.000953935, -0.0118572, 0.000438133, -0.000892954, 0.0208176])
    # Dai
    image_folder = 'data/02_image_after'
    img_filenames = sorted(os.listdir(image_folder),
                           key=lambda x: int(x[:-4]))[23:]
    os.makedirs('data/dai_out', exist_ok=True)
    point_clouds = np.loadtxt(
        'data/02_with_lidar_pos_cloud.txt')[:, :3].reshape(-1, 59, 3)[4:]
    print(point_clouds.shape, len(img_filenames))
    for i, img_filename in enumerate(img_filenames):
        img_filename = os.path.join(image_folder, img_filename)
        img = cv2.imread(img_filename)
        world_points = point_clouds[i]
        camera_points = transformation.lidar_to_camera(
            world_points, extrinsic_matrix)
        pixel_points = transformation.camera_to_pixel(
            camera_points, intrinsic_matrix, distortion_coefficients)
        for x, y in pixel_points:
            x = int(math.floor(x))
            y = int(math.floor(y))
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            cv2.circle(img, (x, y), 3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('data/dai_out/{}.jpg'.format(i + 1), img)
    exit()


def li():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh', action='store_true')
    parser.add_argument('--smooth', action='store_true')
    args = parser.parse_args()

    intrinsic_matrix = np.array([9.8430939171489899e+02, 0., 9.5851460160821068e+02, 0.,
                                 9.8519855566009164e+02, 5.8554990545554267e+02, 0., 0., 1.]).reshape(3, 3)
    extrinsic_matrix = np.array([0.0077827356937, -0.99995676791, 0.0050883521036, -0.0032019276082, 0.0016320898332, -0.0050757970921, -
                                0.99998578618, 0.049557315144, 0.99996838215, 0.007790929719, 0.0015925156874, 0.12791621362, 0, 0, 0, 1]).reshape((4, 4))

    distortion_coefficients = np.array([3.2083739041580001e-01,
                                        2.2269550643173597e-01,
                                        8.8895447057740762e-01,
                                        -2.8404775013002994e+00,
                                        4.867095044851689
                                        ])
    distortion_coefficients = np.zeros(5)

    mocap_to_world_maxtrix = np.array([[2.47890385e-02, -2.97448765e-01, 9.54415898e-01,
                                        1.33231808e+01],
                                       [9.99681638e-01, 2.88194477e-03, -2.50665430e-02,
                                        5.89324659e-02],
                                       [4.70543793e-03, 9.54733436e-01, 2.97425492e-01,
                                        -1.05669274e+00],
                                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                        1.00000000e+00]]).reshape(4, 4)

    C = np.array([[-2.47890308e-02, -9.99681623e-01, -4.70543841e-03,
                   3.95942246e+01],
                  [9.54415933e-01, -2.50665508e-02, 2.97425512e-01,
                   -1.70128328e+01],
                  [-2.97448754e-01, 2.88194721e-03, 9.54733431e-01,
                   8.91624961e+00],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   1.00000000e+00]])

    pc_prefix = '/home/ljl/data/pose_estimation/0817/081701_clustered'
    pc_filenames = sorted(os.listdir(pc_prefix),
                          key=lambda x: int(x[:-4]))
    out_prefix = '/home/ljl/data/pose_estimation/0817/081701_projection'
    os.makedirs(out_prefix, exist_ok=True)

    img_prefix = '/home/ljl/data/pose_estimation/0817/081701_images'
    img_filenames = sorted(os.listdir(img_prefix), key=lambda x: int(x[:-4]))

    valids = []

    if args.bvh:
        mocap_to_lidar_translations = []
        csv_filename = '/home/ljl/data/pose_estimation/0817/wangxin_worldpos.csv'
        mocap_data = mocap.MoCapData(csv_filename)

    for pc_filename in tqdm(pc_filenames):
        pc_index = int(pc_filename[:-4])
        img_index = pc_index * 3 + 1136
        mocap_index = pc_index * 10 + 1453

        pc_filename = os.path.join(pc_prefix, pc_filename)
        img_filename = os.path.join(
            img_prefix, '{:06d}.png'.format(img_index))
        if not os.path.exists(img_filename):
            break

        lidar_points = pcd.read_point_cloud(pc_filename)[:, :3]
        valid = lidar_points.shape[0] > 1
        valids.append(valid)
        if args.bvh:
            mocap_to_lidar_translations.append(transformation.get_mocap_to_lidar_translation(
                mocap_data[mocap_index], lidar_points, C) if valid else None)
            if pc_index == 275:
                constant_translation = transformation.get_mocap_to_lidar_translation(
                    mocap_data[mocap_index], lidar_points, C)

    for i in range(len(mocap_to_lidar_translations)):
        if mocap_to_lidar_translations[i] is None:
            continue
        mocap_to_lidar_translations[i] = constant_translation

    # # smooth
    # if args.bvh and args.smooth:
    #     print('Smooth')
    #     half_width = 10
    #     translation_sum = np.zeros((3, ))
    #     n = len(mocap_to_lidar_translations)
    #     l = 0
    #     r = 0
    #     cnt = 0
    #     aux = []
    #     for i in range(n):
    #         rb = min(n - 1, i + half_width)
    #         lb = max(0, i - half_width)
    #         while r <= rb:
    #             if mocap_to_lidar_translations[r] is not None:
    #                 translation_sum += mocap_to_lidar_translations[r]
    #                 cnt += 1
    #             r += 1
    #         while l < lb:
    #             if mocap_to_lidar_translations[l] is not None:
    #                 translation_sum -= mocap_to_lidar_translations[l]
    #                 cnt -= 1
    #             l += 1
    #         if (mocap_to_lidar_translations[i] is not None) and (cnt > 0):
    #             aux.append(translation_sum / cnt)
    #         else:
    #             aux.append(None)
    #     mocap_to_lidar_translations = aux

    for i, pc_filename in enumerate(tqdm(pc_filenames)):
        # for i, pc_filename in enumerate(pc_filenames):
        #     print(i)
        pc_index = int(pc_filename[:-4])
        img_index = pc_index * 3 + 1136
        mocap_index = pc_index * 10 + 1453

        pc_filename = os.path.join(pc_prefix, pc_filename)
        img_filename = os.path.join(img_prefix, '{:06d}.png'.format(img_index))
        out_img_filename = '{}/{:06d}.jpg'.format(out_prefix, pc_index)
        if not os.path.exists(img_filename):
            break

        if not valids[i]:
            cv2.imwrite(out_img_filename, cv2.imread(img_filename))
            continue

        lidar_points = pcd.read_point_cloud(pc_filename)[:, :3]
        if args.bvh:
            lidar_points = transformation.mocap_to_lidar(
                mocap_data[mocap_index], lidar_points, C, mocap_to_lidar_translations[i])
        pixel_points = transformation.camera_to_pixel(transformation.lidar_to_camera(
            lidar_points, extrinsic_matrix), intrinsic_matrix, distortion_coefficients)

        img = cv2.imread(img_filename)
        for x, y in pixel_points:
            x = int(math.floor(x))
            y = int(math.floor(y))
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            cv2.circle(img, (x, y), 3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(out_img_filename, img)


def zhang():

    from Robosense_object import Calibration

    calib = Calibration('1')
    pc_prefix = 'data/point_cloud'
    pc_filenames = sorted(os.listdir(pc_prefix),
                          key=lambda x: int(x[:-4]))

    # pc_prefix = 'data/2021-07-24-16-02-45-RS-0-Data-10000-12000'
    # pc_filenames = sorted(os.listdir(pc_prefix),
    #                       key=lambda x: int(x[:-4]))[535:]

    os.makedirs('data/0805', exist_ok=True)
    for i, pc_filename in enumerate(pc_filenames):
        pc_filename = os.path.join(pc_prefix, pc_filename)
        # world_points = obj.read_point_cloud(pc_filename)
        world_points = pcd.read_point_cloud(pc_filename)[:, :3]
        pixel_points = calib.project_robo_to_image(world_points)
        pixel_points = pixel_points[:, :2]
        img_filename = 'data/image/{:05d}.jpg'.format(859 + i)
        img = cv2.imread(img_filename)
        for x, y in pixel_points:
            x = int(math.floor(x))
            y = int(math.floor(y))
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            cv2.circle(img, (x, y), 3, color=(0, 0, 255), thickness=-1)
        cv2.imwrite('data/0805/{}.jpg'.format(i + 1), img)
        exit()


if __name__ == '__main__':
    li()
