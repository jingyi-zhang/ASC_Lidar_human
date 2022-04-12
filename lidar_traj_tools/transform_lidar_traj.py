from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
from bvh_tool import Bvh
import sys
import os

def toRt(r, t):
    '''
    将3*3的R转成4*4的R
    '''
    share_vector = np.array([0,0,0,1], dtype=float)[np.newaxis, :]
    r = np.concatenate((r, t.reshape(-1,1)), axis = 1)
    r = np.concatenate((r, share_vector), axis=0)
    return r

def save_in_same_dir(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    np.savetxt(save_file, data, fmt='%.6f')
    print('save file in: ', save_file)

if __name__ == '__main__':
    transfrom_list = []
    if len(sys.argv) < 3:
        print('请输入轨迹文件和旋转矩阵文件')
        lidar_file = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\02lidar\\traj_with_timestamp.txt"
        tfile = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\02lidar\\traj_with_timestamp_first_frame_inv.txt"
        transfrom_list.append(np.loadtxt(tfile, dtype=float))
    else:
        args = len(sys.argv)
        lidar_file = sys.argv[1]
        for i in range(2, args):
            # transfrom_list.append(sys.argv[i])
            transfrom_list.append(np.loadtxt(sys.argv[i], dtype=float))
    lidar = np.loadtxt(lidar_file, dtype=float)
    rots = np.zeros(shape=(lidar.shape[0],4,4))
    
    for t in transfrom_list:
        print(t)


    # 四元数到旋转矩阵
    for i in range(lidar.shape[0]):
        i_frame_R = R.from_quat(lidar[i, 4: 8]).as_matrix()  #3*3
        rots[i] = toRt(i_frame_R, lidar[i, 1:4])   #4*4

    for tt in transfrom_list:
        rots = np.matmul(tt, rots.T).T

    # 旋转矩阵到四元数
    for i in range(lidar.shape[0]):
        lidar[i, 1:4] = rots[i, :3, 3] #平移量
        lidar[i, 4:8] = R.from_matrix(rots[i, :3, :3]).as_quat() #四元数
    # save new traj
    save_in_same_dir(lidar_file, lidar, '_变换后的轨迹')