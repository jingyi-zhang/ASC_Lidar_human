import os
import numpy as np
import open3d as o3d
import tqdm
import glob, json
import matplotlib.pyplot as plt
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--indx', type=str, default='18/1/Fast')
# args = parser.parse_args()


def read_info(indx):
    info_path_ = os.path.join('/hdd24T/zjy/jingyi/gait/xmu_gait/info_json', indx)
    
    r1_path = os.path.join(info_path_,'R1.json')
    with open(r1_path, 'r') as f:
        _r1 = json.load(f)
        r1 = np.asarray(_r1)
    f.close()
    
    r2_path = os.path.join(info_path_,'R2.json')
    with open(r2_path, 'r') as f:
        _r2 = json.load(f)
        r2 = np.asarray(_r2)
    f.close()
    
    points_path = os.path.join(info_path_,'points.json')
    with open(points_path, 'r') as f:
        points = json.load(f)
        x_min = points['x_min']
        x_max = points['x_max']
        y_min = points['y_min']
        y_max = points['y_max']
        _position = points['first_position']
        position = np.asarray(_position)
    f.close()
    
    # r1 = np.array([
    #     0.809960067272,0.402760982513,0.426319301128,1.987177368103,-0.353190660477,
    #     0.915283441544,-0.193681523204,2.460576470795,-0.468210428953,0.006302311085,
    #     0.883594572544,3.437339001732,0.000000000000,0.000000000000,0.000000000000,1.00000000000])
    # r2 = np.array([
    #     -0.409990221262,-0.912089943886,0.000000000000,12.796133041382,0.912089943886,
    #     -0.409990221262,0.000000000000,-15.156480789185,0.000000000000,0.000000000000,
    #     1.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,1.0000000])
    
    # y_min = -8.931463
    # y_max = 3.374676
    # x_min = 5.266186
    # x_max = 13.347358
    # position = np.array([6.839455, -3.877454, -0.718159])
    
    RT = np.dot(r2,r1)
    
    num_particles = 1000  # 粒子数量
    # initial_position = position  # 人体初始位置
    initial_covariance = np.eye(3)  # 初始协方差矩阵
    motion_model_noise = np.diag([0.1, 0.1, 0.1])  # 运动模型噪声协方差c
    # measurement_noise = np.diag([0.1, 0.1, 0.1])  # 测量噪声协方差
    measurement_noise =0.1  # 测量噪声协方差

    return RT, x_min, x_max, y_min, y_max, position, num_particles, initial_covariance,\
           motion_model_noise, measurement_noise


# 定义运动模型
def motion_model(particles, noise):
    # 添加运动模型噪声
    noise_samples = np.random.multivariate_normal(np.zeros(3), noise, num_particles)
    particles += noise_samples
    return particles


# 定义测量模型（根据激光雷达数据更新粒子权重）
def measurement_model(particles, measurement):
    # 计算每个粒子与测量之间的距离
    distances = np.linalg.norm(particles - measurement,axis=1)

    # 使用高斯分布计算权重
    # weights = np.exp(-0.5 * (distances ** 2) / np.diag(measurement_noise))
    weights = np.exp(-0.5 * (distances ** 2) / measurement_noise)
    weights /= np.sum(weights)

    return weights


# 更新粒子权重和估计位置
def update(particles, measurement):
    particles = motion_model(particles, motion_model_noise)
    weights = measurement_model(particles, measurement)
    particles = particles[np.random.choice(num_particles, num_particles, p=weights)]
    estimated_position = np.mean(particles, axis=0)
    return particles, estimated_position


def fix_points_num(points: np.array, num_points: int):
    points = points[~np.isnan(points).any(axis=-1)]

    origin_num_points = points.shape[0]
    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    if origin_num_points >= num_points:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res


def cut_out_person(input_pcds_path, output_pcds_path, RT, x_min, x_max, y_min, y_max, 
                   initial_position, num_particles, initial_covariance, 
                   motion_model_noise, measurement_noise):
        
    particles = np.random.multivariate_normal(initial_position, initial_covariance,num_particles)
    assert os.path.exists(input_pcds_path), 'input_pcds_path不存在！'
    if not os.path.exists(output_pcds_path):
        os.makedirs(output_pcds_path)
    print(f'cut out person from {input_pcds_path} to {output_pcds_path}')

    filenames = glob.glob(os.path.join(input_pcds_path, '*.pcd'))
    filenames.sort(key=lambda e: int(os.path.basename(e).split('.')[0]))

    # outliers = []
    
    persons = []
    estimated_position = initial_position
    for f in tqdm.tqdm(filenames, desc='Remove Ground', ncols=60):
        pcd = o3d.io.read_point_cloud(f)
        points = np.asarray(pcd.points)
        #  put the lidarpoints to plane
        ones = np.ones((points.shape[0],1))
        p_w = np.concatenate((points, ones), axis=1)
        p = np.dot(p_w, RT.T)[:,:3]
        
        y_inds = np.logical_and(y_min < p[:, 1], p[:, 1] < y_max)
        x_inds = np.logical_and(x_min< p[:,0], p[:,0]<x_max)
        indx = np.logical_and(y_inds, x_inds)
        # indx = np.logical_and(0<p[:,2],np.logical_and(y_inds, x_inds))
        p = p[indx]
        pcd.points = o3d.utility.Vector3dVector(p)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        
        cluster_results = np.array(outlier_cloud.cluster_dbscan(eps=0.2, min_points=10, print_progress=False))
        labels = np.unique(cluster_results)
        min_distance = 100
        for label in labels:
            cluster = np.array(outlier_cloud.points)[cluster_results==label]
            cluster_mean = cluster.mean(axis=0)
            distance = np.linalg.norm(cluster_mean - estimated_position)
            if distance < min_distance:
                min_distance = distance
                person = cluster
        fix_points = fix_points_num(np.array(outlier_cloud.points), num_particles)
        particles, estimated_position = update(particles,fix_points)
        persons.append(person)

    for p, f in zip(persons, filenames):
        pcd_file_name = os.path.join(output_pcds_path, f"{os.path.basename(f).split('.')[0]}.pcd")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        o3d.io.write_point_cloud(pcd_file_name, pcd)
    print('Finish!')


if __name__ == '__main__':
    for i in range(94, 105):
        for j in [1,2,3,4]:
            for y in ['Fast', 'Normal', 'Slow']:
                idx = str(i)+'/' + str(j) + '/' + y
                # print(idx)
                # idx = '33/4/Fast'
                pcd_path = os.path.join('/hdd24T/zjy/jingyi/gait/xmu_gait/lidarcap/pointclouds', idx)
                save_path = os.path.join('/hdd24T/zjy/jingyi/gait/xmu_gait/lidarcap/human', idx)
                
                RT, x_min, x_max, y_min, y_max, position, num_particles, \
                initial_covariance, motion_model_noise, measurement_noise = read_info(idx)
                
                cut_out_person(pcd_path, save_path, RT, x_min, x_max, y_min, y_max, 
                               position, num_particles, initial_covariance, 
                               motion_model_noise, measurement_noise)
