#! /usr/bin/env python

from argparse import ArgumentParser
from panoptic_slam.kitti.data_analysers import KittiRawOdomComparator


if __name__ == "__main__":
    parser = ArgumentParser(prog="kitti_data_analysis")

    parser.add_argument("-d", "--kitti_dir", required=True,
                        help="Path to the root of the KITTI dataset (Parent of the sequence/ and raw/ directories).")
    parser.add_argument("-s", "--seq", required=True, type=int,
                        help="KITTI Sequence to compare between Raw and Odometry datasets.")

    args = parser.parse_args()

    analyser = KittiRawOdomComparator(args.kitti_dir, args.seq)
    analyser.compare_velo()
