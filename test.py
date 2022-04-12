import os
import shutil
import sys
import itertools
import numpy as np
sys.path.append('/cwang/home/ljl')
from lidarcap.utils import get_input
# sys.path.append('/home/ljl/ASC_Lidar_human')

from util import path_util, pc_util, img_util, transformation
from io3d import ply
from tqdm import tqdm

# def read_array_dat(filename):
#     import struct
#     array = []
#     with open(filename, 'rb') as f:
#         n = struct.unpack('i', f.read(4))[0]
#         for i in range(n):
#             array.append(struct.unpack('d', f.read(8))[0])
#     return array


# if __name__ == '__main__':
#     indexes = [5, 6, 7, 8, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 38]
#     frames = [len(os.listdir('/xmu_gait/lidarcap/labels/3d/segment/' + str(index)))
#               for index in indexes]
#     rounded_frames = [(frame + 15) // 16 * 16 for frame in frames]
#     rounded_frames.insert(0, 0)
#     rounded_endpoints = list(itertools.accumulate(rounded_frames))
#     dirname = '/home/ljl/lidarcap/visual/run-20211030_135742-35vwvuyc'
#     filenames = path_util.get_sorted_filenames_by_index(
#         os.path.join(dirname, 'pred_vertices_smpl'))
#     print(len(filenames))

#     for i, index in enumerate(indexes):
#         cur_dirname = os.path.join(dirname, str(index))
#         path_util.clear_folder(cur_dirname)
#         old_filenames = filenames[rounded_endpoints[i]                                  :rounded_endpoints[i + 1]]
#         basenames = [os.path.basename(old_filename)
#                      for old_filename in old_filenames]
#         new_filenames = [os.path.join(cur_dirname, basename)
#                          for basename in basenames]
#         for j, (old_filename, new_filename) in enumerate(zip(old_filenames, new_filenames)):
#             if j == frames[i]:
#                 break
#             os.rename(old_filename, new_filename)

#     # timestamps = read_array_dat(
#     #     '/xmu_gait/lidarcap/pointclouds/42/timestamps.dat')
#     # # print(len(timestamps))
#     # print(timestamps[18498] - timestamps[381])


# if __name__ == '__main__':
#     # indexes = [5, 6, 7, 8, 24, 25, 26, 27, 28, 29, 30,
#     #            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

#     indexes = list(range(5, 9)) + list(range(24, 43))

#     for index in indexes:
#         n_segments = len(os.listdir(
#             '/xmu_gait/lidarcap/labels/3d/segment/' + str(index)))
#         n_mocaps = len(
#             np.load('/xmu_gait/lidarcap/mocaps/{}/mocap_indexes.npy'.format(index)))
#         n_images = len(
#             np.load('/xmu_gait/lidarcap/images/{}/image_indexes.npy'.format(index)))
#         print(index, n_segments == n_mocaps and n_mocaps == n_images)

#     print(indexes)
#     exit()

#     indexes = [8, 27, 28]

#     for index in indexes:
#         i = [path_util.get_index(f) for f in path_util.get_sorted_filenames_by_index(
#             '/xmu_gait/lidarcap/labels/3d/segment/' + str(index))]
#         for j in range(1, len(i)):
#             if i[j] - i[j - 1] != 1:
#                 print(j)
#                 break


def crop_image(image, bbox):
    x, y, w, h = bbox
    l = max(w, h)
    return image[y:y + l, x:x + l, :]


def get_data_filenames(idx):
    segment_filenames = path_util.get_sorted_filenames_by_index(
        '/cwang/home/ljl/data/lidarcap/labels/3d/segment/{}'.format(idx))

    image_indexes = np.load(
        '/cwang/home/ljl/data/lidarcap/images/{}/image_indexes.npy'.format(idx))
    start = image_indexes[0]
    rectified_start = start / 30 * 29.83
    diff = start - rectified_start
    image_indexes = image_indexes - diff
    image_indexes = np.around(image_indexes).astype(int)
    # print(image_indexes)
    # exit()
    image_filenames = ['/cwang/home/ljl/data/lidarcap/images/{}/{:06d}.png'.format(
        idx, image_index) for image_index in image_indexes]

    pose_filenames = path_util.get_sorted_filenames_by_index(
        '/cwang/home/ljl/data/lidarcap/labels/3d/pose/{}'.format(idx))
    pose_filenames = list(filter(lambda x: x.endswith('.ply'), pose_filenames))
    return segment_filenames, image_filenames, pose_filenames


if __name__ == '__main__':
    extrinsic_matrix = np.array([-0.0043368991524, -0.99998911867, -0.0017186757713, 0.016471385748, -0.0052925495236, 0.0017416212982, -
                                0.99998447772, 0.080050847871, 0.99997658984, -0.0043277356572, -0.0053000451695, -0.049279053295, 0, 0, 0, 1]).reshape(4, 4)
    intrinsic_matrix = np.array([9.5632709662202160e+02, 0., 9.6209910493679433e+02,
                                0., 9.5687763573729683e+02, 5.9026610775785059e+02, 0., 0., 1.]).reshape(3, 3)
    distortion_coefficients = np.array([-6.1100617222502205e-03, 3.0647823796371827e-02, -
                                       3.3304524444662654e-04, -4.4038460096976607e-04, -2.5974982760794661e-02])

    # image_indexes = np.load(
    #     '/cwang/home/ljl/data/lidarcap/images/7/image_indexes.npy')

    # start = image_indexes[0]
    # rectified_start = start / 30 * 29.83
    # print(start, rectified_start)
    # diff = start - rectified_start
    # image_indexes = image_indexes - diff
    # image_indexes = np.around(image_indexes).astype(int)
    # pc_filenames = path_util.get_sorted_filenames_by_index(
    #     '/cwang/home/ljl/data/lidarcap/labels/3d/segment/7')
    # image_filenames = ['/cwang/home/ljl/data/lidarcap/images/7/{:06d}.png'.format(
    #     image_index) for image_index in image_indexes]
    # output_dir = '/cwang/home/ljl/tmp/projection/7'
    # os.makedirs(output_dir, exist_ok=True)

    idx = 39
    output_folder = '/cwang/home/ljl/tmp/0228/{}'.format(idx)
    os.makedirs(output_folder, exist_ok=True)
    segment_filenames, image_filenames, pose_filenames = get_data_filenames(
        idx)
    print(len(segment_filenames))
    print(len(image_filenames))
    print(len(pose_filenames))
    indexes = [2719, 10143, 10469, 8801, 6210, 3245, 2068, 1560]
    # indexes = [4926, 14817, 12734, 1476, 1641, 6053]
    indexes = sorted(indexes)

    cur = 0
    res = []
    for i in range(len(segment_filenames)):
        if str(indexes[cur]) in segment_filenames[i]:
            res.append(
                (segment_filenames[i], image_filenames[i], pose_filenames[i]))
            shutil.copy(segment_filenames[i], os.path.join(
                output_folder, 'pc_' + os.path.basename(segment_filenames[i])))
            shutil.copy(pose_filenames[i], os.path.join(
                output_folder, 'pose_' + os.path.basename(pose_filenames[i])))

            bbox = get_input.get_bbox(segment_filenames[i])
            image = get_input.read_image(image_filenames[i])
            cropped_image = get_input.crop_image(image, bbox)
            get_input.write_image(os.path.join(
                output_folder, os.path.basename(image_filenames[i])), cropped_image)

            cur += 1
            if cur == len(indexes):
                break

    print(res)

    # for index in indexes:
    #     for i in pc_filenames:
    #         if str(index) in pc_filenames[i]:

    exit()

    for i, (pc_filename, image_filename) in tqdm(enumerate(zip(segment_filenames, image_filenames)), total=len(segment_filenames)):
        lidar_points = ply.read_point_cloud(pc_filename)
        pixel_points = transformation.camera_to_pixel(
            transformation.lidar_to_camera(lidar_points, extrinsic_matrix), intrinsic_matrix, distortion_coefficients)
        pixel_points = pixel_points.astype(int)
        pixel_points = pixel_points[pixel_points[:, 0] > -1]
        pixel_points = pixel_points[pixel_points[:, 0] < 1920]
        pixel_points = pixel_points[pixel_points[:, 1] > -1]
        pixel_points = pixel_points[pixel_points[:, 1] < 1080]
        x_min = pixel_points[:, 0].min()
        y_min = pixel_points[:, 1].min()
        x_length = pixel_points[:, 0].max() - x_min
        y_length = pixel_points[:, 1].max() - y_min
        img_util.project_points_on_image(
            pixel_points, image_filename, '{}/{}.png'.format(output_dir, i + 1))
