import pandas as pd  
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
from bvh_tool import Bvh
import sys
import os

mocap_init = np.array([
    [-1, 0, 0, 0],
    [0, 0, 1, 0], 
    [0, 1, 0, 0], 
    [0, 0, 0, 1]])
# mocap_init = R.from_matrix(mocap_init[:3,:3])

# 时间同步帧
lidar_key = 84 
mocap_key = 554

# mocap的帧率试试lidar的备注
frame_scale = 5 # mocap是100Hz, lidar是20Hz

def toRt(r, t):
    '''
    将3*3的R转成4*4的R
    '''
    share_vector = np.array([0,0,0,1], dtype=float)[np.newaxis, :]
    r = np.concatenate((r, t.reshape(-1,1)), axis = 1)
    r = np.concatenate((r, share_vector), axis=0)
    return r

def world_to_camera(X, K_EX):
    K_inv = np.linalg.inv(K_EX)
    return np.matmul(K_inv[:3,:3], X.T).T + K_inv[:3,3]
    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    focal length / principal point / radial_distortion / tangential_distortion
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2] #focal lendgth
    c = camera_params[..., 2:4] # center principal point
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2

    return f*XXX + c

def save_pose(filepath, poses, skip=100):
    dirname = os.path.dirname(filepath)
    file_name = Path(filepath).stem
    save_file = os.path.join(dirname, file_name + '_cloud.txt')
    shape = poses.shape
    num = np.arange(shape[0])
    pose_num = num.repeat(shape[1]).reshape(shape[0], shape[1], 1)
    poses = np.concatenate((poses, pose_num), -1)  # 添加行号到末尾
    poses_save = poses[num % skip ==0].reshape(-1, 4) #每隔skip帧保存一下
    np.savetxt(save_file, poses_save, fmt='%.6f')
    print('保存pose到: ', save_file)

def render_animation(keypoints, poses, fps, bitrate, azim, output, viewport, cloud=None,
                     limit=-1, downsample=1, size=5, input_video_path=None, input_video_skip=0, frame_num=None, key=None):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """

    path = './data/%s/%s' % (dirs, output.split('.')[0])
    if not os.path.exists(path):
        os.mkdir(path)
    # plt.ioff()
    # figsize = (10, 5)
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    # 2D
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)  # (1,2,1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    # 0, ('Reconstruction', 3d kp)
    for index, (title, data) in enumerate(poses.items()):
        # 3D
        ax = fig.add_subplot(1, 1 + len(poses), index+2,
                             projection='3d')  # (1,2,2)
        ax.view_init(elev=15., azim=azim)
        # set 长度范围
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        # ax.auto_scale_xyz([-radius/2, radius/2], [0, radius], [-radius/2, radius/2])
        # axisEqual3D(ax)
        ax.dist = 9  # 视角距离
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])

        # lxy add
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 轨迹 is base on position 0
        trajectories.append(data[:, 0])  # only add x,y not z
    poses = list(poses.values())

    # 设置2、3的显示
    if cloud is not None:
        for n, ax in enumerate(ax_3d):
            if n == 0:
                norm = plt.Normalize(min(cloud[2]), max(cloud[2]))
                ax.scatter(*cloud, c=norm(cloud[2]),marker='.', s=5, linewidth=0, alpha=0.3, cmap='viridis')
            elif n == 1:
                cloud_in = cloud[:, cloud[2] >= 0.1]
                norm = plt.Normalize(min(cloud_in[2]), max(cloud_in[2]))
                ax.scatter(*cloud_in, c=norm(cloud_in[2]),marker='.', s=3, alpha=0.3, cmap='Greys')
                ax.view_init(elev=90, azim=0)
            elif n == 2:
                cloud_in = cloud[:, cloud[2] < 0.1]
                norm = plt.Normalize(min(cloud_in[2]), max(cloud_in[2]))
                ax.scatter(*cloud_in, c=norm(cloud_in[2]),marker='.', s=3, alpha=0.3, cmap='Greys')
                ax.view_init(elev=0, azim=0)

            if n > 0:
                ax.dist = 7
                middle_y = np.mean(trajectories[n][:, 1])  # y轴的中点
                center = np.mean(cloud_in, -1)  # x的中心
                mb = max(trajectories[n][:, 1]) - min(trajectories[n][:, 1])   # 以轨迹的Y轴为可视化的最大长度
                ax.set_xlim3d([-mb/2 + center[0], mb/2 + center[0]])
                ax.set_ylim3d([-mb/2 + middle_y, mb/2 + middle_y])
                ax.set_zlim3d([-mb/2, mb/2])
    
    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        # 根据kpt长度，决定帧的长度
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        # effective_length = min(keypoints.shape[0], len(all_frames))
        # all_frames = all_frames[:effective_length]

    kpts = keypoints
    initialized = False
    image = None
    lines = []
    numbers = []
    points = None

    if limit < 1:
        limit = len(frame_num)
    else:
        limit = min(limit, len(frame_num))

    vis_enhence = True

    def update_video(i):
        # if i < frame_num[15] or i > frame_num[-15]:
        #     return
        nonlocal initialized, image, lines, points, numbers
        for num in numbers:
            num.remove()
        numbers.clear()
        for n, ax in enumerate(ax_3d):  # 只有1个
            if i > 0:  # 绘制轨迹
                dt = trajectories[n][i-1:i+1]
                # if np.linalg.norm(dt[0] - dt[1]) < 1:
                    # ax.plot(*dt.T, c='red', linewidth=2, alpha=0.5)
            if n == 0:
                ax.set_xlim3d([-radius/2 + 1.2, radius/2 + 1.2])
                ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
                ax.set_ylim3d([-radius/2 + trajectories[n][i, 1],radius/2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius/2 + trajectories[n][i, 2],
                           radius/2 + trajectories[n][i, 2]])
            # axisEqual3D(ax)

        # Update 2D poses
        if not initialized:
            image = ax_in.imshow(all_frames[frame_num[i]], aspect='equal')
            # 画图2D
            points = ax_in.scatter(*kpts[i].T, 2, color='pink', edgecolors = 'white', zorder = 10)
            initialized = True
        else:
            image.set_data(all_frames[frame_num[i]])
            points.set_offsets(keypoints[i])

        if i % 50 == 0 and vis_enhence:
            plt.savefig(
                path + '/' + str(frame_num[i]), dpi=100, bbox_inches='tight')
        print(
            'finish one frame\t {}/{}'.format(frame_num[i], frame_num[-1]), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(
        0, limit), interval=1000/fps, repeat=False)
        
    save_path = os.path.join(path, "%s_%d_%s_%s" %(key, keypoints.shape[0], dirs, output))
    
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(save_path, writer=writer)
    elif output.endswith('.gif'):
        # anim.save(output, dpi=80, writer='imagemagick')
        anim.save(save_path, dpi=80, writer='imagemagick')
    else:
        raise ValueError(
            'Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()

def save_in_same_csv(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.csv')
    data.to_csv(save_file)
    print('save csv in: ', save_file)

def save_in_same_dir(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    np.savetxt(save_file, data, fmt='%.6f')
    print('save traj in: ', save_file)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('请输入 [*pos.csv] [*rot.csv] [*lidar_traj.txt]')
        pospath = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\mocap_csv\\02_with_lidar_pos.csv"
        rotpath = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\mocap_csv\\02_with_lidar_rot.csv"
        lidar_file = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\02lidar\\traj_with_timestamp_变换后的轨迹.txt"
        # in_file = 'sys.argv[4]'
        # ex_file = 'sys.argv[5]'
        # dist_file = 'sys.argv[5]'
    else:
        pospath = sys.argv[1]
        rotpath = sys.argv[2]
        lidar_file = sys.argv[3]
        # in_file = sys.argv[4]
        # ex_file = sys.argv[5]
        # dist_file = sys.argv[5]

    # K_IN = np.loadtxt(in_file, dtype=float)
    # K_EX = np.loadtxt(ex_file, dtype=float)
    # dist_coeff = np.loadtxt(dist_file, dtype=float)

    # 1. 读取数据
    lidar = np.loadtxt(lidar_file, dtype=float)
    pos_data_csv=pd.read_csv(pospath, dtype=np.float32)
    rot_data_csv=pd.read_csv(rotpath, dtype=np.float32)

    pos_data = np.asarray(pos_data_csv) /100 # cm -> m
    mocap_length = pos_data.shape[0]
    pos_data = pos_data[:, 1:].reshape(mocap_length, -1, 3)
    rot_data = np.asarray(rot_data_csv) # 度

    # 2. 输入lidar中对应的帧和mocap中对应的帧, 求得两段轨迹中的公共部分
    lidar_key = lidar_key - int(lidar[0,0]) 
    
    lidar_start = 0
    lidar_end = int(lidar[-1,0]) - int(lidar[0,0]) 
    mocap_start = mocap_key - (lidar_key - lidar_start) * frame_scale
    mocap_end = mocap_key + (lidar_end - lidar_key) * frame_scale

    if (mocap_start) < 0:
        lidar_start = lidar_key - (mocap_key // frame_scale)
        mocap_start = mocap_key % frame_scale 
    if (mocap_end) > mocap_length - 1:
        lidar_end = lidar_key + (mocap_length - 1 - mocap_key) // frame_scale
        mocap_end = mocap_length - 1 - (mocap_length - 1 - mocap_key) % frame_scale

    '''
    3. 将mocap配准到lidar，得到RT，应用于该帧的所有点
    '''
    lidar_first = R.from_quat(lidar[lidar_start, 4: 8]).as_matrix() #第一帧的矩阵
    mocap_first = R.from_euler('yxz', rot_data[mocap_start, 1:4], degrees=True).as_matrix() ##第一帧的旋转矩阵
    mocap_first = np.matmul(mocap_init[:3,:3], mocap_first) #第一帧的旋转矩阵，乘上 mocap坐标系 -> lidar坐标系的变换矩阵

    position = np.zeros(shape=((lidar_end - lidar_start) + 1, pos_data.shape[1], 3))
    new_rot = np.zeros(shape=(position.shape[0], rot_data.shape[1]))
    for i in range(lidar_start, lidar_end + 1):
        # 读取 i 帧的 RT
        R_lidar = R.from_quat(lidar[i, 4: 8]).as_matrix()  #3*3
        R_lidar = np.matmul(R_lidar, np.linalg.inv(lidar_first))
        R_lidar = toRt(R_lidar, lidar[i, 1:4])   #4*4

        # 读取对应 mocap的hip的rt
        mocap_number = (i - lidar_key) * frame_scale + mocap_key # 对应的mocap的帧
        R_mocap = R.from_euler('yxz', rot_data[mocap_number, 1:4], degrees=True).as_matrix() #原始数据
        R_mocap = toRt(R_mocap, pos_data[mocap_number, 0].copy())

        R_mocap = np.matmul(mocap_init, R_mocap) # 变换到lidar坐标系
        R_mocap[:3,:3] = np.matmul(R_mocap[:3, :3], np.linalg.inv(mocap_first)) # 右乘第一帧旋转矩阵的逆

        # 求mocap到Lidar的变换关系
        mocap_to_lidar = np.matmul(R_lidar, np.linalg.inv(R_mocap))

        # 将变换矩阵应用于单帧所有点
        pos_init = np.matmul(mocap_init[:3,:3], pos_data[mocap_number].T) # 3 * m, 先坐标系变换
        position[i] = np.matmul(mocap_to_lidar[:3,:3], pos_init).T + mocap_to_lidar[:3,3] # m * 3，再进行旋转平移

        # 将mocap的所有关节的旋转都改变
        new_rot[i, 0] = rot_data[mocap_number, 0].copy()
        for j in range(rot_data.shape[1]//3):
            R_ij = R.from_euler(
                'yxz', rot_data[mocap_number, j*3 + 1:j*3 + 4], degrees=True).as_matrix()
      
            R_ijj = np.matmul(mocap_init[:3,:3], R_ij)  
            R_ijj = np.matmul(mocap_to_lidar[:3,:3], R_ijj) # mocap->lidar 配准旋转矩阵
            R_ijj = np.matmul(mocap_init[:3,:3], R_ijj)  
            new_rot[i, j*3 + 1:j*3 + 4] = R.from_matrix(R_ijj).as_euler('yxz', degrees=True)
        
    new_rot_csv = pd.DataFrame(new_rot, columns = [col for col in rot_data_csv.columns])
    save_in_same_csv(rotpath, new_rot_csv, '_trans_RT')
    save_in_same_dir(lidar_file, lidar[lidar_start:lidar_end+1], '_与mocap重叠部分') #保存有效轨迹
    # 将3D投影回2D
    # anim_output = {'view1': position}
    # azimuths = 150
    # viz_output = ''
    # viz_bitrate = 400
    # viz_video = "e:\\Daiyudi\\Documents\\OneDrive - stu.xmu.edu.cn\\I_2021_HumanMotion\\数据采集\\0719\\VID_20210722_054704.mp4"
    # render_animation(input_keypoints, anim_output, 20, viz_bitrate,
    #                 azim=azimuths, output=viz_output, limit=300, size= 7, input_video_path=viz_video, frame_num=frame_num, viewport=(2304/2, 1296/2), input_video_skip=0,  key=key)
    # 4. 保存pose
    save_pose(pospath, position, skip = 1)

