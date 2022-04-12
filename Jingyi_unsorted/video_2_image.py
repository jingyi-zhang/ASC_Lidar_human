# -*- coding: utf-8 -*-
# @Author  : jingyi

import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='/mnt/d/human_data/0724/rectangular.mp4')
parser.add_argument('--save_path', type=str, default='/mnt/d/human_data/0724/0724_frame/images')
args = parser.parse_args()

def video2frame(videos_path, frames_save_path, time_interval):
    '''
    :param videos_path: 视频的存放路径
    :param frames_save_path: 视频切分成帧之后图片的保存路径
    :param time_interval: 保存间隔
    :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    print('start')
    while success:
        print(count)
        success, image = vidcap.read()
        count += 1
        if count % time_interval == 0:
            cv2.imencode('.jpg', image)[1].tofile(
                frames_save_path + "/frame%d.jpg" % count)
        # if count == 20:
        #   break
    # print(count)


if __name__ == '__main__':
    videos_path = args.video_path
    frames_save_path = args.save_path
    time_interval = 1
    video2frame(videos_path, frames_save_path, time_interval)