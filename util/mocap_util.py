from util import path_util

import os
import shutil


def get_csvs_from_bvh(bvh_path, mocap_dir):
    assert os.path.isabs(bvh_path) and os.path.isfile(bvh_path)
    assert os.path.isabs(mocap_dir) and os.path.isdir(mocap_dir)
    os.system('bvh-converter -r {}'.format(bvh_path))
    src_dir = os.path.dirname(bvh_path)
    worldpos_csv = path_util.get_one_path_by_suffix(src_dir, '_worldpos.csv')
    rotations_csv = path_util.get_one_path_by_suffix(src_dir, '_rotations.csv')
    shutil.move(worldpos_csv, os.path.join(
        mocap_dir, os.path.basename(worldpos_csv)))
    shutil.move(rotations_csv, os.path.join(
        mocap_dir, os.path.basename(rotations_csv)))
