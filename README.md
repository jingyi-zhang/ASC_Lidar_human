
# 1. bvh_tools
### [从BVH中提取ROOT point](bvh_tools/get_root.py)
```
python get_root.py [bvh_file]
```
- **输入BVH文件，输出hip point的位置**
### [BVH和绝对pose、rot之间的互相转换](https://github.com/OlafHaag/bvh-toolbox)
- 下载工具[bvh-toolbox](https://github.com/OlafHaag/bvh-toolbox)
- 其中的BVH to CSV tables可将bvh转换成绝对坐标的pose
    ```
    bvh2csv [bvhfile] #直接在命令行运行
    ```
# 2. lidar_traj_tools
### [将Mocap的数据配准到LiDAR轨迹到](lidar_traj_tools/mocap_to_lidar.py)
```
python mocap_to_lidar.py [*pos.csv] [*rot.csv] [*lidar_traj.txt]
```
#### 输入
1. Mocap的绝对旋转 [rot.csv] [Mocap坐标系]
2. Mocap的绝对位置 [pos.csv] [Mocap坐标系]
3. lidar轨迹 [点云坐标系]
####  输出
1. 变换后的rot文件 [Mocap坐标系]
2. 变换后的pos文件 [点云坐标系]
3. 公共的轨迹 [点云坐标系]

- ***在代码的17行和18行需输入同步时刻的帧号***
- ***在代码21行需输入Mocap帧率是Lidar帧率的倍数(默认是5)***

## [旋转lidar的轨迹文件](lidar_traj_tools/transform_lidar_traj.py)
```
python transform_lidar_traj.py [lidar_traj_path] [旋转矩阵地址]
```
* 旋转矩阵为4 * 4
* 可输入多个旋转矩阵文件
# 3. visualization
### [利用open3D可视化RT](visualization/visualize_RT.py)
```
python visualize_RT.py -l [lidar_traj_path] # 可视化lidar的轨迹，数据格式是 line_number*9
python visualize_RT.py -b [bvh_path] #可视化bvh文件hip关节的轨迹
python visualize_RT.py -c [csv_pos_path] [csv_rot_path] #可视化rot 和 pos 文件 hip关节的轨迹和旋转
```
* 代码的51-54行指定了点云的位置，可以注释掉或者改成自己的点云文件地址
