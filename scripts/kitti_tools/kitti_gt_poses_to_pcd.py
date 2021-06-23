#! /usr/bin/env python

import argparse
from os import path

from panoptic_slam.kitti.data_loaders import KittiGTPosesLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="kitti_gt_poses_to_pcd")

    parser.add_argument("-d", "--kitti_dir", required=True,
                        help="Path to the root of the KITTI dataset (Parent of the sequence/ and raw/ directories).")
    parser.add_argument("-s", "--seq", required=True, type=int,
                        help="Kitti Sequence to convert the Ground Truth poses from.")
    parser.add_argument("-p", "--pcd_file", required=True,
                        help="Path and filename of the PCD file where to store the converted poses as a pointcloud.")

    args = parser.parse_args()

    kitti_dir = path.expanduser(path.expandvars(args.kitti_dir))
    seq = args.seq
    pcd_file = path.expanduser(path.expandvars(args.pcd_file))

    print("Loading timestamps and ground truth poses for KITTI sequence {}.".format(seq))
    converter = KittiGTPosesLoader(kitti_dir, seq)
    print("Converting poses to PCD file.")
    converter.save_as_pcd(pcd_file)
    print("Ground truth poses saved as PCD file in {}.".format(pcd_file))
