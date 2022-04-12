

import numpy as np


def read_point_cloud(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = list(filter(lambda x: x.startswith('v'), lines))
        point_cloud = []
        for line in lines:
            point = line.split(' ')[1:]
            point = np.array(list(map(float, point)))
            point_cloud.append(point)
        point_cloud = np.stack(point_cloud)
        return point_cloud
