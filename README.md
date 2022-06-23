### 原始数据准备
将pcap, bvh, fbx和MP4放在/SAMSUMG8T/lidarcapv2/raw/xx目录中 
注：fbx和bvh导出的时候需要选取被采集人性别身高，以及数据处理

### 1. pcap文件切帧：
```cd ASC_lidar_human; python utils/pc_utils.py --pcap2pcd --index xx````

### cloudcompare求出mocap到lidar坐标系的外参矩阵
a. 红x、绿：y、蓝：z\
b. 点击左边最上面的立方体\
c. 选择平面，得到rot1\
d. 用剪刀右边的按键（rotation选择z）将点云进行旋转，长边超前，得到rot2\
e. rot = np.dot(rot2, rot1)

### 2. mcap，lidar文件起始帧指定,以及rot指定, box的指定，box是人活动的范围
/SAMSUMG8T/lidarcapv2/raw/xx/process_info.json

### 3. 去背景点云生成
```cd ASC_lidar_human; python util/cut_out_person.py --index xx```

### 4. 生成mocap的csv文件
```python process.py —-index xxx -—log —-gen_basic```

### 5. 生成mocap_indexes
```python process.py --index xxx --log --gen_mocap_indexes```

### 6. 生成smpl
```python process.py --index xxx --log --gen_smpl```

### 7. 生成pose
```python process.py --index 0410 --log --gen_pose```

### 可视化
移步visualization
