#! /usr/bin/env python

from argparse import ArgumentParser
from os import path

from panoptic_slam.io.utils import parse_path, makedirs
from panoptic_slam.kitti.data_analysers.kitti_pose_error import compute_pcd_pose_error

if __name__ == "__main__":

    parser = ArgumentParser(description="Script for evaluating the translational and rotational errors between"
                                        "poses computed by LIO-SAM and stored in a PCD file,"
                                        "and KITTI's Ground Truth poses.")

    parser.add_argument("-d", "--kitti_dir", required=True, type=parse_path,
                        help="Path to the root of the KITTI dataset (Parent of the sequence/ and raw/ directories).")
    parser.add_argument("-s", "--seq", required=True, type=int,
                        help="Kitti Sequence to convert the Ground Truth poses from.")
    parser.add_argument("-p", "--trans_pcd_file", required=True, type=parse_path,
                        help="Path to the PCD File with the trajectory as 6DOF transformations.")

    # TODO: add time_matching, match_alg and n_match_positions arguments

    parser.add_argument("-o", "--output_dir", default=None,
                        help="Path to directory where output files will be saved.")

    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is not None:
        output_dir=parse_path(output_dir)
        makedirs(output_dir)
    else:
        output_dir = path.dirname(args.trans_pcd_file)

    kwargs = {
        'save_path': output_dir
    }

    compute_pcd_pose_error(args.kitti_dir, args.seq, args.trans_pcd_file, **kwargs)
