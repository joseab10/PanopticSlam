#! /usr/bin/env python

import argparse
from os import path

from panoptic_slam.io.utils import parse_path, makedirs
from panoptic_slam.kitti.data_loaders import KittiGTPosesLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="kitti_gt_poses_to_pcd")

    parser.add_argument("-d", "--kitti_dir", required=True, type=parse_path,
                        help="Path to the root of the KITTI dataset (Parent of the sequence/ and raw/ directories).")
    parser.add_argument("-s", "--seq", required=True, type=int,
                        help="Kitti Sequence to convert the Ground Truth poses from.")
    parser.add_argument("-p", "--pcd_file", required=True, type=parse_path,
                        help="Path and filename of the PCD file where to store the converted poses as a pointcloud.")

    args = parser.parse_args()

    makedirs(path.dirname(args.pcd_file))

    print("Loading timestamps and ground truth poses for KITTI sequence {}.".format(args.seq))
    converter = KittiGTPosesLoader(args.kitti_dir, args.seq)
    print("Converting poses to PCD file.")
    converter.save_as_pcd(args.pcd_file)
    print("Ground truth poses saved as PCD file in {}.".format(args.pcd_file))
