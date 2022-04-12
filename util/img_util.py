import cv2
import math
import os
import numpy as np


def video_to_images(video_path, images_dir):
    assert os.path.isabs(video_path) and os.path.isfile(video_path)
    assert os.path.isabs(images_dir) and os.path.isdir(images_dir)
    os.system(
        'ffmpeg -i {} -r 29.83 -f image2 -v error -s 1920x1080 {}/%06d.png'.format(video_path, images_dir))


def images_to_videos(video_path, images_dir):
    assert os.path.isabs(video_path) and os.path.isfile(video_path)
    assert os.path.isabs(images_dir) and os.path.isdir(images_dir)
    os.system('ffmpeg -y -threads 16 -i {}/%06d.png -profile:v baseline -level 3.0 -c:v libx264 -pix_fmt yuv420p -an -v error {}'.format(images_dir, video_path))


def project_points_on_image(points, img_filename, out_img_filename):
    img = cv2.imread(img_filename)
    for x, y in points:
        x = int(math.floor(x))
        y = int(math.floor(y))
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            continue
        cv2.circle(img, (x, y), 1, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(out_img_filename, img)
