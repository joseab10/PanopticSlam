#! /usr/bin/env python

from argparse import ArgumentParser
from os import path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np

from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder
from panoptic_slam.io.utils import parse_path, mkdir


def rotmat2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotate(a, angle):
    return np.matmul(rotmat2d(angle), a.T).T


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for plotting the paths from KITTI's Ground Truth.")

    parser.add_argument("-d", "--kitti_dir", required=True, type=parse_path,
                        help="Path to the root of the KITTI dataset (Parent of the sequence/ and raw/ directories).")

    parser.add_argument("-o", "--output_dir", type=parse_path,
                        help="Path to directory where output files will be saved.")
    parser.add_argument("-x", "--ext", type=str, default="pdf",
                        help="(str, Default=pdf) Extension of the output plot files.")

    parser.add_argument("-s", "--label_step", type=int, default=25,
                        help="(int, Default: 25) Label poses only at multiples of step.")
    parser.add_argument("-r", "--label_radius", type=float, default=20.0,
                        help="(float, Default: 20) Radius at which the label will be away from "
                        "its corresponding point.")
    parser.add_argument("-t", "--label_theta", type=float, default=-3.0 * np.pi / 4.0,
                        help="(float, Default: -3pi/4) Direction of the label with respect to "
                             "the travel distance at the corresponding point.")

    args = parser.parse_args()
    output_dir = args.output_dir
    mkdir(output_dir)

    plt.rcParams['figure.figsize'] = [28.00, 14.0]
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "CMU Serif"
    plt.ioff()

    radius = args.label_radius
    th = args.label_theta
    step = args.label_step
    extension = args.ext

    for s in range(11):
        kitti = KittiOdomDataYielder(args.kitti_dir, s)

        poses = kitti.get_poses()
        poses = poses[:, ::2, 3:].reshape((-1, 2))

        labels = np.arange(len(poses))
        labels_mask = labels % step == 0
        labels = labels[labels_mask]

        txt_poses = poses[labels_mask]

        # Compute the gradient of the trajectory to know where to put the labels
        pgrad = np.diff(poses, axis=0)
        pgrad_norm = np.sqrt(np.sum(np.square(pgrad), axis=1))
        pgrad_norm[pgrad_norm == 0] = 1.0  # To avoid division by zero
        pgrad_norm = pgrad_norm.reshape((-1, 1))
        pgrad = pgrad / pgrad_norm  # Normalize the gradient to a unit-length vector
        pgrad = np.vstack([pgrad[[0]], pgrad])  # So that the first point has the same gradient as the next one
        pgrad = pgrad[labels_mask]  # Slice it so that only the gradient for the points of interest remain

        tmp_pos_inc = rotate(pgrad, th)

        # Code for modifying the labels that happen to fall close or on top of other labels
        txt_pos = txt_poses + radius * tmp_pos_inc

        pgrad_grad_x = np.square(txt_pos[:, 0] - txt_pos[:, 0, np.newaxis])
        pgrad_grad_y = np.square(txt_pos[:, 1] - txt_pos[:, 1, np.newaxis])
        pgrad_grad = np.sqrt(pgrad_grad_x + pgrad_grad_y)
        pgrad_grad = np.tril(pgrad_grad)
        pgrad_grad = pgrad_grad < (radius / 2)
        pgrad_grad = np.logical_and(pgrad_grad,  np.tril(np.ones_like(pgrad_grad)))
        pgrad_grad = np.logical_and(pgrad_grad, np.logical_not(np.diag(np.ones_like(pgrad_grad[0]))))
        pgrad_grad_counts = np.sum(pgrad_grad, axis=1)

        inv_th_msk = pgrad_grad_counts > 0
        txt_pos[inv_th_msk] = txt_poses[inv_th_msk] + radius * rotate(pgrad[inv_th_msk], -th)

        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_axisbelow(True)
        plt.grid(which="minor", axis="both", color="black", linestyle="--", linewidth=0.25)
        plt.grid(which="major", axis="both", color="black", linestyle="--", linewidth=0.375)

        ax.scatter(poses[labels_mask, 0], poses[labels_mask, 1],
                   s=2.5, c="black", marker="s", linewidths=0.0)
        ax.scatter(poses[np.logical_not(labels_mask), 0], poses[np.logical_not(labels_mask), 1],
                   s=0.875, c="black", marker="o", linewidths=0.0)

        for label, point, label_pos in zip(labels, txt_poses, txt_pos):
            ax.annotate(label, point, xytext=label_pos, fontsize=7,
                        arrowprops={'arrowstyle': "->",
                                    'linewidth': 0.5
                                    }
                        )
        min_lim = np.min(poses, axis=0) - 1.5 * radius
        max_lim = np.max(poses, axis=0) + 1.5 * radius
        ax.set_aspect("equal")
        ax.set_xlim([min_lim[0], max_lim[0]])
        ax.set_ylim([min_lim[1], max_lim[1]])
        ax.set_title("KITTI Sequence {:02d} Ground Truth Trajectory".format(s))
        ax.set_xlabel("x\n[m]")
        ax.set_ylabel("z\n[m]", rotation=0)

        fig_file = "trajectory_{:02d}.{}".format(s, extension)
        plt.savefig(path.join(output_dir, fig_file))
        print("Trajectory figure saved in {}.".format(fig_file))
