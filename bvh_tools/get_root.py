from pathlib import Path
import numpy as np
import sys
import os
from bvh_tool import Bvh
from scipy.spatial.transform import Rotation as R


def loadjoint(frames, joint_number, frame_time=0.0333333):
    head = frames[:, joint_number * 6:joint_number*6+3] / 100
    frame_number = np.arange(frames.shape[0])
    frame_number = frame_number.reshape((-1, 1))
    frame_time = frame_number * frame_time

    '''
    从mocap的坐标系方向，转到velodyne坐标系方向，保证Z轴朝上
    '''
    rz = R.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix() # 绕Z转180
    rx = R.from_rotvec(270 * np.array([1, 0, 0]), degrees=True).as_matrix() # 绕X转270°
    init_rot = np.matmul(rx, rz) #先绕Z转180°，再绕X转270°
    head = np.matmul(head, init_rot.T)

    # head = head[:, (0, 2, 1)]  # 交换y z 位置
    # head[:,0] *= -1 # x取负值

    head = np.concatenate((head, frame_number, frame_time), axis=-1)
    return head


# 读取文件
if len(sys.argv) == 2:
    joints_file = sys.argv[1]
else:
    joints_file = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021上科大交流\\演示数据\\0712\\test_0712_ASC_corridor.bvh"

with open(joints_file) as f:
    mocap = Bvh(f.read())

#读取数据
frame_time = mocap.frame_time
frames = mocap.frames
frames = np.asarray(frames, dtype='float32')

head = loadjoint(frames, 0, frame_time)  # 单位：米

# 保存文件
dirname = os.path.dirname(joints_file)
# file_name = os.path.basename(joints_file)
file_name = Path(joints_file).stem
save_file = os.path.join(dirname, file_name + '_root.txt')
np.savetxt(save_file, head, fmt="%.3f")
print('save root in: ', save_file)
#python get_root.py [csv_file_path]