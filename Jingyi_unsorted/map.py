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
import pcl
import argparse
import os, sys
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pcd_file_path', type=str, 
                    default='/mnt/c/Users/Administrator/Desktop/map/lidar',
                    help='the path contains pcd files')
parser.add_argument('--save_file', type=str,
                    default='map.pcd')
args = parser.parse_args()

def save_pcd(points, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    handle = open(save_path, 'a')
    
    point_num = points.shape[0]
    
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')
    
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(
            points[i, 2])
        handle.write(string)
    handle.close()
    
    
if __name__ == '__main__':
    icp = pcl.IterativeClosestPoint()
    Root_path = args.pcd_file_path
    save_path = args.save_file
    for _,_,files in os.walk(Root_path):
        pcd_num = len(files)
    
        lidar_0 = os.path.join(Root_path, '1.pcd')
        scan0=pcl.load_XYZI(lidar_0).to_array()[:, :3]
        final_scan = scan0
        scan0_pc=pcl.PointCloud(scan0)
    
        for indx in tqdm(range(95, 110)):
            filename = str(indx)+'.pcd'
            file_path = os.path.join(Root_path, filename)
            scan = pcl.load_XYZI(file_path).to_array()[:, :3]
            ones = np.ones(len(scan))
            new_scan = np.matrix(np.column_stack((scan, ones))).T
            scan_pc = pcl.PointCloud(scan)
            T = icp.icp(scan_pc, scan0_pc)
            T = np.matrix(T[1])
            scan2scan0=np.dot(T, new_scan)[0:3].T
            final_scan = np.vstack((final_scan,scan2scan0))
        save_pcd(final_scan, save_path)
