#! /usr/bin/env python

import argparse

import rosbag

from panoptic_slam.kitti.converters import RawKitti2LioSamSeqRosBagConverter
from panoptic_slam.ros.utils import parse_rosbag_compression

_conversions = [
    "convert_static_tf",
    "convert_dynamic_tf",
    "convert_imu",
    "convert_raw_imu",
    "convert_gps_fix",
    "convert_gps_vel",
    "convert_cameras",
    "convert_velodyne"
]


def parse_conversions(conv_arg):
    conversions = conv_arg.split(',')
    conversions = ["convert_" + c for c in conversions]
    return {c: c in conversions for c in _conversions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="RawKitti2LioSAMSeqRosBAG")

    parser.add_argument("-p", "--kitti_path", type=str,
                        help="Path to KITTI Dataset directory. \
                             Inside this directory must be the /sequences and /raw directories.")
    parser.add_argument("-s", "--seq", type=int,
                        help="KITTI Odometry dataset sequence. Valid values: (0-21).")
    parser.add_argument("-o", "--output", type=str,
                        help="Path and filename to output ROSBag.")
    parser.add_argument("-c", "--convert", type=parse_conversions, default='static_tf,raw_imu,gps_fix,gps_vel,velodyne',
                        help="Comma-separated list of data to be converted. Can include one or more of \
                        [static_tf, dynamic_tf, imu, raw_imu, gps-fix, gps_vel, cameras]")
    parser.add_argument("--compression", type=parse_rosbag_compression, default="none", choices=["none", "bz2", "lz4"],
                        help="Compression algorithm for storing the output ROSBag.")

    args, unknown_args = parser.parse_known_args()

    unknown_args[:-1:2] = [a.replace("-", "") for a in unknown_args[:-1:2]]  # Remove dashes for unknown args
    kwargs = dict(zip(unknown_args[:-1:2], unknown_args[1::2]))
    kwargs.update(args.convert)

    bag = rosbag.Bag(args.output, "w", compression=args.compression)

    converter = RawKitti2LioSamSeqRosBagConverter(bag, kitti_dir=args.kitti_path, seq=args.seq, **kwargs)

    try:
        converter.convert()
        print("\n\nConversion Successful!\nBag Summary:")
        print(bag)
    finally:
        bag.close()
