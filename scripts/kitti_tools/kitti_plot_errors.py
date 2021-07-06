#! /usr/bin/env python

from argparse import ArgumentParser
from itertools import cycle

from os import path

import numpy as np
import matplotlib.pyplot as plt

from panoptic_slam.io.utils import parse_path, mkdir


def plot_size(size, ignore_chars="[(]) ", sep=","):

    tmp_size = size
    # Ignore characters
    for c in ignore_chars:
        tmp_size = size.replace(c, "")

    size_list = tmp_size.split(sep)  # Split values
    size_list = filter(None, size_list)  # Remove empty elements

    size_list = [float(s) for s in size_list]

    if len(size_list) == 1:
        return [size_list[0], size_list[0]]

    if len(size_list) == 2:
        return size_list

    raise ValueError("Invalid plot size ({}). Two floating values separated by {} required: w{}h.".format(
        size, sep, sep))


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-f", "--error_files", nargs="+", required=True, type=parse_path,
                        help="Path to the error csv file.")
    parser.add_argument("-l", "--legends", nargs="+", required=True,
                        help="Title to apply to the curves from each of the files")
    parser.add_argument("-o", "--output_dir", required=True, type=parse_path, default=".",
                        help="Path to the directory where the plots will be saved.")
    parser.add_argument("-n", "--num_curves", type=int, default=5,
                        help="Number of curves to be plotted from each file.")
    parser.add_argument("-a", "--average_frames", type=int, default=50,
                        help="Average the pose errors")
    parser.add_argument("-t", "--plot_title", type=str, default="",
                        help="Title to be prepended to the figures.")
    parser.add_argument("-s", "--plot_size", type=plot_size, default=[9.6, 7.2],
                        help="Size of the figures as w,h in inches")
    parser.add_argument("-d", "--dpi", type=float, default=100,
                        help="Resolution of the figures in DPI.")

    parser.add_argument("-p", "--do_plot", action="store_true",
                        help="Plot the errors and save the figures as svg.")
    parser.add_argument("-c", "--save_csv", action="store_true",
                        help="Save the aggregated data to csv files, for plotting directly in LaTex for example.")

    args = parser.parse_args()

    if not (args.do_plot or args.save_csv):
        print("Nothing to do then. Use options \"-p\" and/or \"-c\".\n")
        parser.print_help()
        exit(0)

    num_files = len(args.error_files)
    num_legends = len(args.legends)

    if num_files != num_legends:
        raise ValueError("Number of files ({}) must match number of legend strings ({}).".format(
            num_files, num_legends))

    step_column = 0
    frame_column = 1
    err_columns = [4, 5]
    used_columns = [step_column, frame_column]
    used_columns.extend(err_columns)

    plt.ioff()
    tra_err_fig = plt.figure(1, figsize=args.plot_size, dpi=args.dpi)
    tra_err_ax = tra_err_fig.add_subplot(1, 1, 1)
    rot_err_fig = plt.figure(2, figsize=args.plot_size, dpi=args.dpi)
    rot_err_ax = rot_err_fig.add_subplot(1, 1, 1)

    avg_frames = args.average_frames

    tra_legends = []
    rot_legends = []

    colors = [plt.cm.get_cmap(cmap)(np.linspace(0, 1, args.num_curves + 1))
              for cmap in ["Blues_r", "Reds_r", "Purples_r", "Oranges_r"]]

    legend_locations = ["upper left", "lower right", "upper right", "lower left"]

    mkdir(args.output_dir)

    for f, legend_title, cmap, leg_loc in zip(args.error_files, args.legends, cycle(colors), cycle(legend_locations)):
        errors = np.loadtxt(f)
        errors = errors[:, used_columns]  # Discard unused columns

        steps = np.max(errors[:, step_column])
        curve_steps = int((np.round(steps / args.num_curves)))

        start_frame = int(np.min(errors[:, 1]))
        end_frame = int(np.max(errors[:, 1]))

        start_x = np.arange(start_frame, end_frame + 1, step=avg_frames)
        end_x = start_x + avg_frames - 1

        if start_x[-1] >= end_frame:
            start_x = start_x[:-1]
            end_x = end_x[:-1]

        if end_x[-1] > end_frame:
            end_x[-1] = end_frame

        err_steps = errors[:, 0]

        tra_plots = []
        rot_plots = []
        steps_legend = []

        err_file = np.empty((0, 7))

        for i in range(args.num_curves):
            ss = i * curve_steps + 1
            se = (i + 1) * curve_steps + 1

            curve_err = errors[np.logical_and(err_steps >= ss, err_steps <= se)]

            curve_frames = curve_err[:, 1]
            curve_err = curve_err[:, 2:]

            step_frame_min = np.min(curve_frames)
            step_frame_max = np.max(curve_frames)

            step_frame_starts = start_x[np.logical_and(start_x >= step_frame_min, start_x < step_frame_max)]
            step_frame_ends = step_frame_starts + avg_frames

            if step_frame_ends[-1] > step_frame_max:
                step_frame_ends[-1] = step_frame_max

            frame_err = [np.abs(curve_err[np.logical_and(curve_frames >= fs, curve_frames <= fe)])
                         for fs, fe in zip(step_frame_starts, step_frame_ends)]

            frame_avg_err = np.array([np.mean(frame_errors, axis=0) for frame_errors in frame_err])
            frame_sdv_err = np.array([np.std(frame_errors, axis=0) for frame_errors in frame_err])

            if args.do_plot:
                tra_err_ax.fill_between(step_frame_ends, frame_avg_err[:, 0] - frame_sdv_err[:, 0],
                                        frame_avg_err[:, 0] + frame_sdv_err[:, 0], alpha=0.5, color=cmap[i])
                tra_plot, = tra_err_ax.plot(step_frame_ends, frame_avg_err[:, 0], color=cmap[i])
                tra_plots.append(tra_plot)

                rot_err_ax.fill_between(step_frame_ends, frame_avg_err[:, 1] - frame_sdv_err[:, 1],
                                        frame_avg_err[:, 1] + frame_sdv_err[:, 1], alpha=0.25, color=cmap[i])
                rot_plot, = rot_err_ax.plot(step_frame_ends, frame_avg_err[:, 1], color=cmap[i])
                rot_plots.append(rot_plot)

                steps_legend.append("{}-{}".format(ss, se))

            if args.save_csv:
                err_curve_table = np.hstack([np.ones_like(step_frame_ends).reshape((-1, 1)) * i,
                                             step_frame_starts.reshape((-1, 1)), step_frame_ends.reshape((-1, 1)),
                                             frame_avg_err[:, [0]], frame_sdv_err[:, [0]],
                                             frame_avg_err[:, [1]], frame_sdv_err[:, [1]]])
                err_file = np.append(err_file, err_curve_table, axis=0)

        if args.do_plot:
            tra_legend = tra_err_ax.legend(tra_plots, steps_legend, title=legend_title + " (Steps)", loc=leg_loc)
            tra_legends.append(tra_legend)
            rot_legend = rot_err_ax.legend(rot_plots, steps_legend, title=legend_title + " (Steps)", loc=leg_loc)
            rot_legends.append(rot_legend)

        if args.save_csv:
            file_path = path.join(args.output_dir, legend_title + "_agg_err.csv")
            np.savetxt(file_path, err_file, comments="#", header=" ".join(["step",
                                                                           "step_first_frame", "step_last_frame",
                                                                           "tra_err_mean_[m]", "tra_err_sdev_[m]",
                                                                           "rot_err_mean[rad]", "rot_err_sdev_[rad]"]))
            print("Saved errors to: {}.".format(file_path))

    if args.do_plot:
        tra_title = "Translational Error"
        rot_title = "Rotational Error"
        if args.plot_title:
            tra_title = args.plot_title + " (" + tra_title + ")"
            rot_title = args.plot_title + " (" + rot_title + ")"

        tra_err_ax.set_title(tra_title)
        tra_err_ax.set_xlabel("Frames")
        tra_err_ax.set_ylabel(r"$\varepsilon_t$ [m]")

        for leg in tra_legends[:-1]:
            # Re-insert the legends for all but the last file, which should still be there
            tra_err_ax.add_artist(leg)

        rot_err_ax.set_title(rot_title)
        rot_err_ax.set_xlabel("Frames")
        rot_err_ax.set_ylabel(r"$\varepsilon_r$ [rad]")

        for leg in rot_legends[:-1]:
            # Re-insert the legends for all but the last file, which should still be there
            rot_err_ax.add_artist(leg)

        tra_err_file = path.join(args.output_dir, "tra_err.svg")
        tra_err_fig.savefig(tra_err_file)
        print("Saved plot to: {}.".format(tra_err_file))

        rot_err_file = path.join(args.output_dir, "rot_err.svg")
        rot_err_fig.savefig(rot_err_file)
        print("Saved plot to: {}.".format(rot_err_file))

    if args.save_csv:
        abc = 3
