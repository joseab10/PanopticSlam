#!/usr/bin/env python

from argparse import ArgumentParser

from panoptic_slam.io.utils import parse_path
from panoptic_slam.kitti.data_loaders import download_raw_data


if __name__ == "__main__":
    parser = ArgumentParser(prog="raw_kitti_download")

    parser.add_argument("download_dir", type=parse_path,
                        help="Directory where the ZIP data will be downloaded")
    parser.add_argument("-e", "--extract_dir", required=False, type=parse_path,
                        help="Directory where ZIP files will be extracted."
                             "If left undefined, files will be extracted to DOWNLOAD_DIR.")
    subparsers = parser.add_subparsers(title='Dataset download by')
    seq_parser = subparsers.add_parser("s", help="Download by Sequence. Arguments:\n\t * SEQ: Sequence Number")
    seq_parser.add_argument("seq", type=int, help="KITTI Odometry Sequence. \
    Data will be downloaded for corresponding Raw Date and Drive.")
    dd_parser = subparsers.add_parser("d", help="Download by Date and Drive. Arguments:\
    \n\t * DATE: Drive Date\n\t * DRIVE: Drive Number")
    dd_parser.add_argument("date", help="Raw Data Drive Date. Must be in YYYY_MM_DD format")
    dd_parser.add_argument("drive", type=int, help="Raw Data Drive Number.")

    parser.add_argument("-x", "--extract", action="store_true",
                        help="Files will be extracted from ZIP Files if set.")
    parser.add_argument("-r", "--remove_zip", action="store_true",
                        help="ZIP files will be deleted after extraction if set.")

    subsets = ["sync", "extract", "track", "calib"]
    default_subsets = "sync,extract,calib"
    parser.add_argument("-s", "--subsets", default=default_subsets,
                        help="Comma-separated list of data sub-sets to download. Possible elements: [" +
                             ", ".join(subsets) + "]. Default: \"" + default_subsets + "\".")

    args = parser.parse_args()

    kwargs = {k.replace("-", ""): v for k, v in vars(args).items()}
    del kwargs['download_dir']
    if kwargs['extract_dir'] is None:
        del kwargs['extract_dir']
    dl_subsets = {"dl_" + k: True for k in args.subsets.split(",") if k in subsets}
    kwargs.update(dl_subsets)

    download_raw_data(args.download_dir, **kwargs)
