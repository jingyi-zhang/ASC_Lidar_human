# -*- coding: utf-8 -*-
# @Author  : xuelun
import os
import cv2
import numpy as np
import sys


scale = 5
size = 50
if len(sys.argv) < 2:
    print('请输入图片路径')
    exit(-1)
imgPath = sys.argv[1]
# imgPath = 'G:\CamareCalibarate\corrida_1\pick.jpg'
img = cv2.imread(imgPath)
origin = img
c = width = img.shape[2]
w = width = img.shape[1]
h = width = img.shape[0]
print("Image size:", w, h, c)

list_xy = []


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        list_xy.append([x, y])
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow('image', img)
    if event == cv2.EVENT_MOUSEMOVE:
        xy = "%d,%d" % (x, y)
        ymin = max(0, y-size)
        ymax = min(h, y+size)
        xmin = max(0, x-size)
        xmax = min(w, x+size)
        xsize = (xmax - xmin) * scale
        ysize = (ymax - ymin) * scale

        area = origin[ymin:ymax, xmin:xmax].copy()
        area = cv2.resize(area, dsize=(xsize, ysize),
                          interpolation=cv2.INTER_LINEAR)
        cv2.circle(area, ((x-xmin) * scale, (y-ymin) * scale),
                   3, (255, 0, 0), thickness=-1)
        cv2.putText(area, xy, ((x-xmin) * scale, (y-ymin) * scale), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
        cv2.imshow('area', area)


cv2.namedWindow('image')
cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)
cv2.imshow('image', img)

while (cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 0):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindows()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

imgdir = os.path.dirname(imgPath)
np.savetxt(os.path.join(imgdir, 'list_2d.txt'), np.array(list_xy), fmt='%d')
cv2.imwrite(os.path.join(imgdir, 'picked.jpg'), img)
