#! /usr/bin/env python

from argparse import ArgumentParser
from os import path

import numpy as np
import pypcd
from scipy.spatial.transform import Rotation

from panoptic_slam.io.utils import parse_path, mkdir


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for transforming a .PCD point cloud file"
                                        " using a transformation matrix.")

    parser.add_argument("input_pcd_file", type=parse_path,
                        help="Path to the input point cloud file.")
    parser.add_argument("transform_file", type=parse_path,
                        help="Path to the transformation matrix file.")
    parser.add_argument("output_pcd_file", type=parse_path,
                        help="Path where the output, transformed point cloud file will be saved.")

    parser.add_argument("-i", "--invert", action="store_true",
                        help="Invert the transformation matrix if set before applying it.")

    args = parser.parse_args()

    tf = np.eye(4)
    # Ignore first column as it is the step from LIO-SAM, reshape back to 2D matrix form, and place it in the 4x4 matrix
    tf[:3, :] = np.loadtxt(args.transform_file)[1:].reshape((3, 4))
    # Invert if desired
    if args.invert:
        tf = np.linalg.inv(tf)

    print("Transform Matrix loaded from: {}.".format(args.transform_file))

    in_cloud = pypcd.point_cloud_from_path(args.input_pcd_file)
    print("Point Cloud loaded from: {}.".format(args.input_pcd_file))

    x = in_cloud.pc_data['x'].view(np.float32)
    y = in_cloud.pc_data['y'].view(np.float32)
    z = in_cloud.pc_data['z'].view(np.float32)
    t = np.vstack([x, y, z, np.ones_like(x)]).T

    if all(f in in_cloud.fields for f in ["roll", "pitch", "yaw"]):
        # Poses are 6dof
        roll = in_cloud.pc_data['roll'].view(np.float32)
        pitch = in_cloud.pc_data['pitch'].view(np.float32)
        yaw = in_cloud.pc_data['yaw'].view(np.float32)

        euler = np.vstack([roll, pitch, yaw]).T

        r = Rotation.from_euler("xyz", euler).as_dcm()

        p = np.zeros((x.shape[0], 4, 4))
        p[:, :3, :3] = r
        p[:, :, 3:] = t.reshape((-1, 4, 1))

        new_p = np.matmul(tf, p)

        new_r = new_p[:, :3, :3]
        new_euler = Rotation.from_dcm(new_r).as_euler("xyz")
        new_t = new_p[:, :3, 3:]

        in_cloud.pc_data['roll'] = new_euler[:, 0].ravel()
        in_cloud.pc_data['pitch'] = new_euler[:, 1].ravel()
        in_cloud.pc_data['yaw'] = new_euler[:, 2].ravel()

    else:
        new_t = np.matmul(tf, t.T).T
        new_t = new_t[:, :3]

    in_cloud.pc_data['x'] = new_t[:, 0].ravel()
    in_cloud.pc_data['y'] = new_t[:, 1].ravel()
    in_cloud.pc_data['z'] = new_t[:, 2].ravel()

    mkdir(path.dirname(args.output_pcd_file))
    in_cloud.save(args.output_pcd_file)
    print("\nTransformed PCD file saved at: {}.".format(args.output_pcd_file))
