import sys
sys.path.append('/cwang/home/mqh/ASC_Lidar_Human')

import argparse

from server import run

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', type=str, default='1111')
parser.add_argument('-i', '--index', type=str, required=True)
args = parser.parse_args()

#run(args.port, args.index)
run(args.index, args.index)