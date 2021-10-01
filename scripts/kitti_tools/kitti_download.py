#!/usr/bin/env python

from argparse import ArgumentParser

from panoptic_slam.io.utils import parse_path
from panoptic_slam.kitti.data_loaders import download_data


if __name__ == "__main__":
    parser = ArgumentParser(prog="raw_kitti_download")

    # Main Arguments
    parser.add_argument("download_dir", type=parse_path,
                        help="Directory where the ZIP data will be downloaded")
    dataset_choices = ["raw", "odom"]
    parser.add_argument("dataset", choices=dataset_choices,
                        help="Dataset to be downloaded. Valid values: {}".format(",".join(dataset_choices)))

    # Raw KITTI Arguments
    parser.add_argument("--seq", type=int, default=None,
                        help="KITTI Odometry Sequence. Data will be downloaded for corresponding Raw Date and Drive.")
    parser.add_argument("--date", default=None,
                        help="Raw Data Drive Date. Must be in YYYY_MM_DD format")
    parser.add_argument("--drive", type=int, default=None,
                        help="Raw Data Drive Number.")

    default_raw_subsets = {
        'sync':    True,
        'extract': True,
        'track':   False,
        'calib':   True,
    }

    for k, v in default_raw_subsets.items():
        parser.add_argument("--dl_" + k, type=bool, default=v,
                            help="Download Raw's {} subset. Default: {}.".format(k, v))

    # KITTI Odometry Arguments
    default_odom_subsets = {
        'calib': True,
        'odo':   True,
        'velo':  True,
    }
    for k, v in default_odom_subsets.items():
        parser.add_argument("--dl_" + k, type=bool, default=v,
                            help="Download Odometry's {} subset. Default: {}.".format(k, v))

    # Download, extract and remove Arguments
    parser.add_argument("--max_processes", type=int, default=None,
                        help="Number of parallel processes to run when downloading multiple files."
                             "Default: None.")
    parser.add_argument("--chunk_size", type=int, default=256*1024,
                        help="Size of the chunks in bytes in which the file will be downloaded."
                             "Default: 262'144 (256Kb)")
    parser.add_argument("-e", "--extract_dir", type=parse_path,
                        help="Directory where ZIP files will be extracted."
                             "If left undefined, files will be extracted to DOWNLOAD_DIR.")
    parser.add_argument("-x", "--extract", action="store_true",
                        help="Files will be extracted from ZIP Files if set.")
    parser.add_argument("-r", "--remove_zip", action="store_true",
                        help="ZIP files will be deleted after extraction if set.")

    args = parser.parse_args()

    kwargs = {k.replace("-", ""): v for k, v in vars(args).items()}

    download_data(args.download_dir, args.dataset, **kwargs)
