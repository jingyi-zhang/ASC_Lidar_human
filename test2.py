import json
import os
# import shutil
# import sys
# import itertools
# import numpy as np
# sys.path.append('/home/ljl/ASC_Lidar_human')

from util import path_util, pc_util, img_util, transformation
from io3d import pcd
# from tqdm import tqdm

# # def read_array_dat(filename):
# #     import struct
# #     array = []
# #     with open(filename, 'rb') as f:
# #         n = struct.unpack('i', f.read(4))[0]
# #         for i in range(n):
# #             array.append(struct.unpack('d', f.read(8))[0])
# #     return array


# # if __name__ == '__main__':
# #     indexes = [5, 6, 7, 8, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 38]
# #     frames = [len(os.listdir('/xmu_gait/lidarcap/labels/3d/segment/' + str(index)))
# #               for index in indexes]
# #     rounded_frames = [(frame + 15) // 16 * 16 for frame in frames]
# #     rounded_frames.insert(0, 0)
# #     rounded_endpoints = list(itertools.accumulate(rounded_frames))
# #     dirname = '/home/ljl/lidarcap/visual/run-20211030_135742-35vwvuyc'
# #     filenames = path_util.get_sorted_filenames_by_index(
# #         os.path.join(dirname, 'pred_vertices_smpl'))
# #     print(len(filenames))

# #     for i, index in enumerate(indexes):
# #         cur_dirname = os.path.join(dirname, str(index))
# #         path_util.clear_folder(cur_dirname)
# #         old_filenames = filenames[rounded_endpoints[i]                                  :rounded_endpoints[i + 1]]
# #         basenames = [os.path.basename(old_filename)
# #                      for old_filename in old_filenames]
# #         new_filenames = [os.path.join(cur_dirname, basename)
# #                          for basename in basenames]
# #         for j, (old_filename, new_filename) in enumerate(zip(old_filenames, new_filenames)):
# #             if j == frames[i]:
# #                 break
# #             os.rename(old_filename, new_filename)

# #     # timestamps = read_array_dat(
# #     #     '/xmu_gait/lidarcap/pointclouds/42/timestamps.dat')
# #     # # print(len(timestamps))
# #     # print(timestamps[18498] - timestamps[381])


# # if __name__ == '__main__':
# #     # indexes = [5, 6, 7, 8, 24, 25, 26, 27, 28, 29, 30,
# #     #            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

# #     indexes = list(range(5, 9)) + list(range(24, 43))

# #     for index in indexes:
# #         n_segments = len(os.listdir(
# #             '/xmu_gait/lidarcap/labels/3d/segment/' + str(index)))
# #         n_mocaps = len(
# #             np.load('/xmu_gait/lidarcap/mocaps/{}/mocap_indexes.npy'.format(index)))
# #         n_images = len(
# #             np.load('/xmu_gait/lidarcap/images/{}/image_indexes.npy'.format(index)))
# #         print(index, n_segments == n_mocaps and n_mocaps == n_images)

# #     print(indexes)
# #     exit()

# #     indexes = [8, 27, 28]

# #     for index in indexes:
# #         i = [path_util.get_index(f) for f in path_util.get_sorted_filenames_by_index(
# #             '/xmu_gait/lidarcap/labels/3d/segment/' + str(index))]
# #         for j in range(1, len(i)):
# #             if i[j] - i[j - 1] != 1:
# #                 print(j)
# #                 break


# if __name__ == '__main__':
# #     print(np.load('./29.npy',allow_pickle=True))
#     extrinsic_matrix = np.array([-0.0043368991524, -0.99998911867, -0.0017186757713, 0.016471385748, -0.0052925495236, 0.0017416212982, -
#                                 0.99998447772, 0.080050847871, 0.99997658984, -0.0043277356572, -0.0053000451695, -0.049279053295, 0, 0, 0, 1]).reshape(4, 4)
#     intrinsic_matrix = np.array([9.5632709662202160e+02, 0., 9.6209910493679433e+02,
#                                 0., 9.5687763573729683e+02, 5.9026610775785059e+02, 0., 0., 1.]).reshape(3, 3)
#     distortion_coefficients = np.array([-6.1100617222502205e-03, 3.0647823796371827e-02, -
#                                        3.3304524444662654e-04, -4.4038460096976607e-04, -2.5974982760794661e-02])

#     image_indexes = np.load('/xmu_gait/lidarcap/images/29/image_indexes.npy')

#     start = image_indexes[0]
#     rectified_start = start / 30 * 29.83
#     print(start, rectified_start)
#     diff = start - rectified_start
#     image_indexes = image_indexes - diff
#     image_indexes = np.around(image_indexes).astype(int)
#     pc_filenames = path_util.get_sorted_filenames_by_index(
#         '/xmu_gait/lidarcap/labels/3d/segment/29')
#     image_filenames = [
#         '/xmu_gait/lidarcap/images/29/{:06d}.jpg'.format(image_index) for image_index in image_indexes]
#     output_dir = '/home/ljl/tmp/projection/29'
#     os.makedirs(output_dir, exist_ok=True)
#     a = {}
#     a[1] = {}
#     c = []
#     a[1]['frames'] = np.array([x for x in range(0,len(image_filenames)-1)])
#     for i, (pc_filename, image_filename) in tqdm(enumerate(zip(pc_filenames, image_filenames)), total=len(pc_filenames)):
#         lidar_points = ply.read_point_cloud(pc_filename)
#         pixel_points = transformation.camera_to_pixel(
#             transformation.lidar_to_camera(lidar_points, extrinsic_matrix), intrinsic_matrix, distortion_coefficients)
#         res = img_util.project_points_on_image(
#             pixel_points, image_filename, '{}/{}.jpg'.format(output_dir, i + 1))
#         a[1]['bbox'].append(res)
#     np.save('./29.npy',a)

from collections import Counter
from sklearn.cluster import DBSCAN

import numpy as np

# 聚类


def dbscan_outlier_removal(points):
    labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(points)
    print(Counter(labels))
    return points[labels != -1], points[labels == 1]
    # for label in sort_labels:
    #     cur_points = points[labels == label]
    #     # 初始的聚类，人是站直的，所以可以根据z的高度来把不是人的排除
    #     if pre_center is None:
    #         z_min = min(cur_points[:, 2])
    #         z_max = max(cur_points[:, 2])
    #         if z_max - z_min < 1.4:
    #             print('too low!')
    #             exit()
    #             continue
    #         return points[labels == label]
    #     # 保证人是连续移动的，相邻两帧的位置相差不超过一定阈值
    #     else:
    #         cur_center = np.mean(cur_points, axis=0)
    #         if np.linalg.norm(cur_center - pre_center) > 0.5:
    #             continue
    #         return points[labels == label]
    # return points[0].reshape(1, -1)


for i in range(24, 39):
    points = pcd.read_point_cloud('/cwang/home/ljl/tmp/29_38.pcd')
    reserved, removed = dbscan_outlier_removal(points)
    pcd.save_point_cloud('/cwang/home/ljl/tmp/29_38_reserved.pcd', reserved)
    pcd.save_point_cloud('/cwang/home/ljl/tmp/29_38_removed.pcd', removed)
    # print(start_index)
    # first_frame_points =
