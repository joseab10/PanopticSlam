from collections import namedtuple
import datetime as dt
from os import path
import re

import cv2
import numpy as np

from exceptions import KittiError, KittiGTError
from panoptic_slam.io import FPathTree
from timestamps import KittiTimestamps
from panoptic_slam.geometry.transforms import utils as tu


class KittiConfig:

    # Named tuples for tabular configuration data
    _KittiRawSeqMap = namedtuple("KittiRawSeqMap", "date, drive, start_frame, end_frame")
    _KittiString = namedtuple("KittiString", "types, arg, fmt, validation, re")
    _KittiFile = namedtuple("KittiFile", "node, parser")
    OxtsPacket = namedtuple(
        'OxtsPacket',
        'lat, lon, alt, ' +
        'roll, pitch, yaw, ' +
        'vn, ve, vf, vl, vu, ' +
        'ax, ay, az, af, al, au, ' +
        'wx, wy, wz, wf, wl, wu, ' +
        'pos_accuracy, vel_accuracy, ' +
        'navstat, numsats, ' +
        'posmode, velmode, orimode')

    # Configuration
    # ========================================================================================
    # Mapping between KITTI Raw Drives and KITTI Odometry Sequences
    _raw_seq_mapping = {
        # KEY|                                   |       | Frame | Frame
        # Seq|                            Date   | Drive | Start | End
        # ---+-----------------------------------+-------+-------+------
        0:  _KittiRawSeqMap(dt.date(2011, 10,  3),     27,      0, 4540),
        1:  _KittiRawSeqMap(dt.date(2011, 10,  3),     42,      0, 1100),
        2:  _KittiRawSeqMap(dt.date(2011, 10,  3),     34,      0, 4660),
        3:  _KittiRawSeqMap(dt.date(2011,  9, 26),     67,      0, 800),
        4:  _KittiRawSeqMap(dt.date(2011,  9, 30),     16,      0, 270),
        5:  _KittiRawSeqMap(dt.date(2011,  9, 30),     18,      0, 2760),
        6:  _KittiRawSeqMap(dt.date(2011,  9, 30),     20,      0, 1100),
        7:  _KittiRawSeqMap(dt.date(2011,  9, 30),     27,      0, 1100),
        8:  _KittiRawSeqMap(dt.date(2011,  9, 30),     28,   1100, 5170),
        9:  _KittiRawSeqMap(dt.date(2011,  9, 30),     33,      0, 1590),
        10: _KittiRawSeqMap(dt.date(2011,  9, 30),     34,      0, 1200),
    }

    # Camera Configuration
    _raw_camera_cfg = {
        'GRAY': {
            'L': 0,
            'R': 1,
        },
        'COLOR': {
            'L': 2,
            'R': 3,
        }
    }

    # KITTI Strings formatting and validation
    _seq_fmt_str = "{seq:02d}"
    _seq_val_str = r"([01][0-9])|(2[01])"
    _dat_fmt_str = "{date.year:04d}_{date.month:02d}_{date.day:02d}"
    _dat_val_str = r"(2011)_((09_((2[689])|(30)))|(10_03))"
    _drv_fmt_str = "{drive:04d}"
    _drv_val_str = r"0[0-9]{3}"
    _cam_fmt_str = "{cam:02d}"
    _cam_val_str = r"0[0-3]"
    _f06_fmt_str = "{frame:06d}"
    _f06_val_str = r"[0-9]{6}"
    _f10_fmt_str = "{frame:010d}"
    _f10_val_str = r"[0-9]{10}"

    _str_format_dict = {
        #           |                 Allowed   | Format|    Format   |  Validation  | Compiled
        #    KEY    |                  Types    |  Arg  |    String   |     REGEX    |  REGEX
        # ----------+---------------------------+-------+-------------+--------------+---------
        'seq':        _KittiString(        [int],   "seq", _seq_fmt_str, _seq_val_str, None),
        'date':       _KittiString([dt.datetime],  "date", _dat_fmt_str, _dat_val_str, None),
        'drive':      _KittiString(        [int], "drive", _drv_fmt_str, _drv_val_str, None),
        'raw_camera': _KittiString(        [int],   "cam", _cam_fmt_str, _cam_val_str, None),
        'raw_image':  _KittiString(        [int], "frame", _f10_fmt_str, _f10_val_str, None),
        'raw_oxts':   _KittiString(        [int], "frame", _f10_fmt_str, _f10_val_str, None),
        'rawe_velo':  _KittiString(        [int], "frame", _f10_fmt_str, _f10_val_str, None),
        'raws_velo':  _KittiString(        [int], "frame", _f10_fmt_str, _f10_val_str, None),
        'odo_labels': _KittiString(        [int], "frame", _f06_fmt_str, _f06_val_str, None),
        'odo_velo':   _KittiString(        [int], "frame", _f06_fmt_str, _f06_val_str, None),
    }

    # KITTI Raw Dataset download url
    _urls = {
        "raw": {
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
        },
        "odo": {
            'calib': "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
            'velo': "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip",
            'odom': "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip",
        }
    }

    # Kitti Raw available datasets
    _raw_datasets = {
        '2011_09_26': {
            # City
            1:   ['extract', 'sync', 'tracklets'],
            2:   ['extract', 'sync', 'tracklets'],
            5:   ['extract', 'sync', 'tracklets'],
            9:   ['extract', 'sync', 'tracklets'],
            11:  ['extract', 'sync', 'tracklets'],
            13:  ['extract', 'sync', 'tracklets'],
            14:  ['extract', 'sync', 'tracklets'],
            17:  ['extract', 'sync', 'tracklets'],
            18:  ['extract', 'sync', 'tracklets'],
            48:  ['extract', 'sync', 'tracklets'],
            51:  ['extract', 'sync', 'tracklets'],
            56:  ['extract', 'sync', 'tracklets'],
            57:  ['extract', 'sync', 'tracklets'],
            59:  ['extract', 'sync', 'tracklets'],
            60:  ['extract', 'sync', 'tracklets'],
            84:  ['extract', 'sync', 'tracklets'],
            91:  ['extract', 'sync', 'tracklets'],
            93:  ['extract', 'sync', 'tracklets'],
            95:  ['extract', 'sync'],
            96:  ['extract', 'sync'],
            104: ['extract', 'sync'],
            106: ['extract', 'sync'],
            113: ['extract', 'sync'],
            117: ['extract', 'sync'],
            # Residential
            19:  ['extract', 'sync', 'tracklets'],
            20:  ['extract', 'sync', 'tracklets'],
            22:  ['extract', 'sync', 'tracklets'],
            23:  ['extract', 'sync', 'tracklets'],
            35:  ['extract', 'sync', 'tracklets'],
            36:  ['extract', 'sync', 'tracklets'],
            39:  ['extract', 'sync', 'tracklets'],
            46:  ['extract', 'sync', 'tracklets'],
            61:  ['extract', 'sync', 'tracklets'],
            64:  ['extract', 'sync', 'tracklets'],
            79:  ['extract', 'sync', 'tracklets'],
            86:  ['extract', 'sync', 'tracklets'],
            87:  ['extract', 'sync', 'tracklets'],
            # Road
            15:  ['extract', 'sync', 'tracklets'],
            27:  ['extract', 'sync', 'tracklets'],
            28:  ['extract', 'sync', 'tracklets'],
            29:  ['extract', 'sync', 'tracklets'],
            32:  ['extract', 'sync', 'tracklets'],
            52:  ['extract', 'sync', 'tracklets'],
            70:  ['extract', 'sync', 'tracklets'],
            101: ['extract', 'sync'],
            # Calibration
            119: ['extract'],
        },
        '2011_09_28': {
            # City
            2:   ['extract', 'sync'],
            # Campus
            16:  ['extract', 'sync'],
            21:  ['extract', 'sync'],
            34:  ['extract', 'sync'],
            35:  ['extract', 'sync'],
            37:  ['extract', 'sync'],
            38:  ['extract', 'sync'],
            39:  ['extract', 'sync'],
            43:  ['extract', 'sync'],
            45:  ['extract', 'sync'],
            47:  ['extract', 'sync'],
            # Person
            53:  ['extract', 'sync'],
            54:  ['extract', 'sync'],
            57:  ['extract', 'sync'],
            65:  ['extract', 'sync'],
            66:  ['extract', 'sync'],
            68:  ['extract', 'sync'],
            70:  ['extract', 'sync'],
            71:  ['extract', 'sync'],
            75:  ['extract', 'sync'],
            77:  ['extract', 'sync'],
            78:  ['extract', 'sync'],
            80:  ['extract', 'sync'],
            82:  ['extract', 'sync'],
            86:  ['extract', 'sync'],
            87:  ['extract', 'sync'],
            89:  ['extract', 'sync'],
            90:  ['extract', 'sync'],
            94:  ['extract', 'sync'],
            95:  ['extract', 'sync'],
            96:  ['extract', 'sync'],
            98:  ['extract', 'sync'],
            100: ['extract', 'sync'],
            102: ['extract', 'sync'],
            103: ['extract', 'sync'],
            104: ['extract', 'sync'],
            106: ['extract', 'sync'],
            108: ['extract', 'sync'],
            110: ['extract', 'sync'],
            113: ['extract', 'sync'],
            117: ['extract', 'sync'],
            119: ['extract', 'sync'],
            121: ['extract', 'sync'],
            122: ['extract', 'sync'],
            125: ['extract', 'sync'],
            126: ['extract', 'sync'],
            128: ['extract', 'sync'],
            132: ['extract', 'sync'],
            134: ['extract', 'sync'],
            135: ['extract', 'sync'],
            136: ['extract', 'sync'],
            138: ['extract', 'sync'],
            141: ['extract', 'sync'],
            143: ['extract', 'sync'],
            145: ['extract', 'sync'],
            146: ['extract', 'sync'],
            149: ['extract', 'sync'],
            153: ['extract', 'sync'],
            154: ['extract', 'sync'],
            155: ['extract', 'sync'],
            156: ['extract', 'sync'],
            160: ['extract', 'sync'],
            161: ['extract', 'sync'],
            162: ['extract', 'sync'],
            165: ['extract', 'sync'],
            166: ['extract', 'sync'],
            167: ['extract', 'sync'],
            168: ['extract', 'sync'],
            171: ['extract', 'sync'],
            174: ['extract', 'sync'],
            177: ['extract', 'sync'],
            179: ['extract', 'sync'],
            183: ['extract', 'sync'],
            184: ['extract', 'sync'],
            185: ['extract', 'sync'],
            186: ['extract', 'sync'],
            187: ['extract', 'sync'],
            191: ['extract', 'sync'],
            192: ['extract', 'sync'],
            195: ['extract', 'sync'],
            198: ['extract', 'sync'],
            199: ['extract', 'sync'],
            201: ['extract', 'sync'],
            204: ['extract', 'sync'],
            205: ['extract', 'sync'],
            208: ['extract', 'sync'],
            209: ['extract', 'sync'],
            214: ['extract', 'sync'],
            216: ['extract', 'sync'],
            220: ['extract', 'sync'],
            222: ['extract', 'sync'],
            # Calibration
            225: ['extract'],
        },
        '2011_09_29': {
            # City
            26:  ['extract', 'sync'],
            71:  ['extract', 'sync'],
            # Road
            4:   ['extract', 'sync'],
            # Calibration
            108: ['extract'],
        },
        '2011_09_30': {
            # Residential
            18:  ['extract', 'sync'],
            20:  ['extract', 'sync'],
            27:  ['extract', 'sync'],
            28:  ['extract', 'sync'],
            33:  ['extract', 'sync'],
            34:  ['extract', 'sync'],
            # Road
            16:  ['extract', 'sync'],
            # Calibration
            72:  ['extract'],
        },
        '2011_10_03': {
            # Residential
            27:  ['extract', 'sync'],
            34:  ['extract', 'sync'],
            # Road
            42:  ['extract', 'sync'],
            47:  ['extract', 'sync'],
            # Calibration
            58:  ['extract'],
        },
    }

    _semantic_labels = {
        0: "unlabeled",
        1: "outlier",

        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other vehicle",

        30: "person",
        31: "bicyclist",
        32: "motorcyclist",

        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other ground",

        50: "building",
        51: "fence",
        52: "other structure",
        60: "lane marking",

        70: "vegetation",
        71: "trunk",
        72: "terrain",

        80: "pole",
        81: "traffic sign",
        99: "other object",

        252: "moving car",
        253: "moving bicyclist",
        254: "moving person",
        255: "moving motorcyclist",
        256: "moving on-rails",
        257: "moving bus",
        258: "moving truck",
        259: "moving other vehicle"
    }

    # KITTI Directory Structure
    # File Pointers (declared individually to retain a reference to them)
    _odo_odo_pos = FPathTree(_seq_fmt_str + ".txt", desc="odo_odo_poses")
    _odo_sem_pos = FPathTree("poses.txt")
    _odo_cal = FPathTree("calib.txt")
    _odo_tim = FPathTree("times.txt")
    _odo_lbl = FPathTree(_f06_fmt_str + ".label", desc="labels")
    _odo_vel = FPathTree(_f06_fmt_str + ".bin", desc="velo_data")

    _raw_cal_cam2cam = FPathTree("calib_cam_to_cam.txt")
    _raw_cal_imu2vel = FPathTree("calib_imu_to_velo.txt")
    _raw_cal_vel2cam = FPathTree("calib_velo_to_cam.txt")

    _raw_ext_cam_tim = FPathTree("timestamps.txt")
    _raw_ext_cam_dat = FPathTree(_f10_fmt_str + ".png", desc="cam_data")
    _raw_ext_oxt_tim = FPathTree("timestamps.txt")
    _raw_ext_oxt_dat = FPathTree(_f10_fmt_str + ".txt")
    _raw_ext_vel_tim = FPathTree("timestamps.txt")
    _raw_ext_vel_tim_sta = FPathTree("timestamps_start.txt")
    _raw_ext_vel_tim_end = FPathTree("timestamps_end.txt")
    _raw_ext_vel_dat = FPathTree(_f10_fmt_str + ".txt")

    _raw_syn_cam_tim = FPathTree("timestamps.txt")
    _raw_syn_cam_dat = FPathTree(_f10_fmt_str + ".png", desc="cam_data")
    _raw_syn_oxt_tim = FPathTree("timestamps.txt")
    _raw_syn_oxt_dat = FPathTree(_f10_fmt_str + ".txt", desc="oxts_data")
    _raw_syn_vel_tim = FPathTree("timestamps.txt")
    _raw_syn_vel_tim_sta = FPathTree("timestamps_start.txt")
    _raw_syn_vel_tim_end = FPathTree("timestamps_end.txt")
    _raw_syn_vel_dat = FPathTree(_f10_fmt_str + ".txt", desc="velo_data")

    # Directory Structure
    _kitti_dir_structure = FPathTree("", desc="dataset_root", children=[
        FPathTree("odometry_poses", children=[
            _odo_odo_pos,
        ]),

        FPathTree("sequences", children=[
            FPathTree("{seq:02d}", desc="seq", children=[
                _odo_cal,
                _odo_sem_pos,
                _odo_tim,
                FPathTree("labels", children=[
                    _odo_lbl,
                ]),
                FPathTree("velodyne", children=[
                    _odo_vel,
                ]),
            ]),
        ]),

        FPathTree("raw", children=[
            FPathTree(_dat_fmt_str, desc="date_dir", children=[
                _raw_cal_cam2cam,
                _raw_cal_imu2vel,
                _raw_cal_vel2cam,
                FPathTree(_dat_fmt_str + "_drive_" + _drv_fmt_str + "_extract", desc="drive_ext", children=[
                    FPathTree("image_{cam:02d}", desc="cam", children=[
                        _raw_ext_cam_tim,
                        FPathTree("data", children=[
                            _raw_ext_cam_dat,
                        ])
                    ]),
                    FPathTree("oxts", children=[
                        FPathTree("data_format.txt"),
                        _raw_ext_oxt_tim,
                        FPathTree("data", children=[
                            _raw_ext_oxt_dat,
                        ]),
                    ]),
                    FPathTree("velodyne_points", children=[
                        _raw_ext_vel_tim,
                        _raw_ext_vel_tim_sta,
                        _raw_ext_vel_tim_end,
                        FPathTree("data", children=[
                            _raw_ext_vel_dat,
                        ]),
                    ]),
                ]),
                FPathTree(_dat_fmt_str + "_drive_" + _drv_fmt_str + "_sync", desc="drive_syn", children=[
                    FPathTree("image_{cam:02d}", desc="cam", children=[
                        _raw_syn_cam_tim,
                        FPathTree("data", children=[
                            _raw_syn_cam_dat,
                        ])
                    ]),
                    FPathTree("oxts", children=[
                        FPathTree("data_format.txt"),
                        _raw_syn_oxt_tim,
                        FPathTree("data", children=[
                            _raw_syn_oxt_dat,
                        ]),
                    ]),
                    FPathTree("velodyne_points", children=[
                        _raw_syn_vel_tim,
                        _raw_syn_vel_tim_sta,
                        _raw_syn_vel_tim_end,
                        FPathTree("data", children=[
                            _raw_syn_vel_dat,
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ])

    # Transformations
    _tf_frames = [
        "base_link", "imu", "velo", "cam00", "cam01", "cam02", "cam03"
    ]
    _pose_frame = "cam00"  # TODO: Double check

    # Based on the drawing from: http://www.cvlibs.net/datasets/kitti/setup.php
    _tf_baselink_to_imu = tu.transform_from_rot_trans(np.eye(3), np.array([-2.71/2.0-0.05, 0.32, 0.93]))

    @staticmethod
    def parse_floats_timestamp_str(ts_str):
        """
        Parses a string representing a timestamp as a floating point in seconds.

        :param ts_str: (str) String representing a timestamp as a float in seconds.

        :return: (float) Timestamp in nanoseconds.
        """
        return float(ts_str) * 1e9

    @staticmethod
    def parse_datetime_timestamp_str(ts_str, **kwargs):
        """
        Parses a string representing a timestamp as a date and time.

        :param ts_str: (str) String representing a timestamp as a float in seconds.
        :param kwargs: * format (str, Default: "%Y-%m-%d %H:%M:%S.%f") format in which the date-time is represented.

        :return: (float) Timestamp in nanoseconds.
        """
        fmt = kwargs.get("format") or '%Y-%m-%d %H:%M:%S.%f'
        epoch = dt.datetime.utcfromtimestamp(0)
        ts = dt.datetime.strptime(ts_str[:-4], fmt)
        ts = (ts - epoch).total_seconds() * 1e9
        return ts

    @staticmethod
    def parse_timestamp_file(filepath, ts_parser, **kwargs):

        in_range = kwargs.get("in_range") or (lambda x: True)
        offset = kwargs.get("offset") or 0

        if not callable(in_range):
            raise TypeError("Invalid in_range parameter. It must be a callable that returns a bool.")

        with open(filepath, 'r') as f:
            ts_list = [ts_parser(line) + offset for i, line in enumerate(f) if in_range(i)]

        timestamps = KittiTimestamps(ts_list)

        return timestamps

    @classmethod
    def parse_raw_timestamp_file(cls, filepath, **kwargs):
        return cls.parse_timestamp_file(filepath, cls.parse_floats_timestamp_str, **kwargs)

    @classmethod
    def parse_odo_timestamp_file(cls, filepath, **kwargs):
        return cls.parse_timestamp_file(filepath, cls.parse_datetime_timestamp_str, **kwargs)

    @staticmethod
    def _parse_calib_value(value):
        try:
            return np.array([float(x) for x in value.split()])
        except ValueError:
            pass

        try:
            return dt.datetime.strptime(value, "%d-%b-%Y %H:%M:%")
        except ValueError:
            pass

        raise ValueError("Unsuported calibration data {}".format(value))

    @classmethod
    def parse_calib_file(cls, filepath, **_):
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                data[key] = cls._parse_calib_value(value)

        return data

    @classmethod
    def parse_image(cls, img_path, **kwargs):
        cv_image = cv2.imread(img_path)

        camera = kwargs["cam"]

        if cls.camera_is_grayscale(camera):
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        return cv_image

    @classmethod
    def parse_calib_cam2cam(cls, filepath, **kwargs):
        cam2cam_calib = cls.parse_calib_file(filepath, **kwargs)
        data = {}

        direct_copy_keys = ["corner_dist", "calib_time"]

        for k in direct_copy_keys:
            if k in cam2cam_calib:
                data[k] = cam2cam_calib[k]

        cameras = [c for c in (cls._raw_camera_cfg[k].keys() for k in cls._raw_camera_cfg.keys())]

        matrix_data_shapes = {
            "S": None,         # Unrectified image size (1x2)
            "K": (3, 3),       # Unrectified Intrinsic Matrix (3x3)
            "D": None,         # Unrectified Camera Distortion Vector (1x5)
            "R": (3, 3),       # Unrectified Extrinsic Rotation Matrix (3x3)
            "T": None,         # Unrectified Extrinsic Translation Vector (1x3)
            "S_rect": None,    # Rectified image size (1x2)
            "R_rect": (3, 3),  # Rectified Rotation Matrix to make image planes co-planar (3x3)
            "P_rect": (3, 4)   # Rectified Projection Matrix (3x4)
        }

        for c in cameras:
            cam_key = cls.format_raw_camera(c)
            data[c] = {}

            # Add the reshaped calibration matrices
            for data_key, data_shape in matrix_data_shapes.items():
                calib_key = data_key + "_" + cam_key

                if matrix_data_shapes[data_key] is None:
                    data[c][data_key] = cam2cam_calib[calib_key]
                else:
                    data[c][data_key] = cam2cam_calib[calib_key].reshape(data_shape)
        return data

    @classmethod
    def parse_calib_imu2vel(cls, filepath, **kwargs):
        calib_imu2vel = cls.parse_calib_file(filepath, **kwargs)

        # Reshape matrices
        calib_imu2vel['R'] = calib_imu2vel['R'].reshape((3, 3))  # Rotation Matrix (3x3)

        return calib_imu2vel

    @classmethod
    def parse_calib_vel2cam(cls, filepath, **kwargs):
        calib_vel2cam = cls.parse_calib_file(filepath, **kwargs)

        calib_vel2cam['R'] = calib_vel2cam['R'].reshape((3, 3))

        return calib_vel2cam

    @classmethod
    def parse_calib_odom(cls, filepath, **kwargs):
        calib = cls.parse_calib_file(filepath, **kwargs)

        matrix_data_shapes = {
            "P0": (3, 4),
            "P1": (3, 4),
            "P2": (3, 4),
            "P3": (3, 4),
            "Tr": (3, 4),
        }

        for k, shape in matrix_data_shapes.items():
            calib[k] = calib[k].reshape(shape)

        return calib

    @classmethod
    def parse_velo_bin(cls, filepath, **_):
        scan = np.fromfile(filepath, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        return scan

    @classmethod
    def parse_velo_txt(cls, filepath, **_):
        scan = np.loadtxt(filepath, dtype=np.float32)
        return scan

    @classmethod
    def _raise_no_gt(cls, **kwargs):
        if "seq" not in kwargs:
            raise KittiError("Sequence required for labels.")

        if not cls.has_gt(kwargs['seq']):
            raise KittiGTError("Sequence {} does not have a ground truth.".format(kwargs['seq']))

    @classmethod
    def parse_labels(cls, filepath, **kwargs):

        cls._raise_no_gt(**kwargs)

        panoptic_labels = np.fromfile(filepath, dtype=np.int32)
        semantic_labels = np.bitwise_and(panoptic_labels, 0xFFFF)
        instance_labels = np.right_shift(panoptic_labels, 16)

        return panoptic_labels, semantic_labels, instance_labels

    @classmethod
    def parse_poses(cls, filepath, **kwargs):

        cls._raise_no_gt(**kwargs)

        poses = np.loadtxt(filepath, dtype=np.float32)
        as_tf_matrix = kwargs.get("as_tf_matrix", True)

        if not as_tf_matrix:
            poses = poses.reshape((-1, 3, 4))
            orientations = poses[:, :, 0:3]
            positions = poses[:, :, 3]
            return positions, orientations

        # Add projection vector to complete (4x4) homogeneous transformation matrix
        p_vector = np.zeros((poses.shape[0], 4))
        p_vector[:, -1] = 1
        poses = np.hstack([poses, p_vector])
        poses = poses.reshape((-1, 4, 4))

        return poses

    @classmethod
    def parse_oxts(cls, filepath, **_):
        with open(filepath, 'r') as f:
            lines = f.read()

        if len(lines) != 1:
            raise KittiError("Invalid OXTS file ({}).".format(filepath))

        oxts = lines[0].split()

        oxts[:-5] = [float(x) for x in oxts[:-5]]
        oxts[-5:] = [int(float(x)) for x in oxts[-5:]]

        return cls.OxtsPacket(*oxts)

    # Reference Dict to important files
    _kitti_file_dict = {
        # KEY                                              Path Node   | ParserF
        # ------------------------------+------------------------------+--------
        # Odometry Dataset
        "odo_odo_pose":                 _KittiFile(        _odo_odo_pos, parse_poses),
        "odo_sem_pose":                 _KittiFile(        _odo_sem_pos, parse_poses),
        "odo_calib":                    _KittiFile(            _odo_cal, parse_calib_odom),
        "odo_times":                    _KittiFile(            _odo_tim, parse_odo_timestamp_file),
        "odo_labels":                   _KittiFile(            _odo_lbl, parse_labels),
        "odo_velo":                     _KittiFile(            _odo_vel, parse_velo_bin),
        # Raw Calibration Files
        "raw_calib_cam2cam":            _KittiFile(    _raw_cal_cam2cam, parse_calib_cam2cam),
        "raw_calib_imu2vel":            _KittiFile(    _raw_cal_imu2vel, parse_calib_imu2vel),
        "raw_calib_vel2cam":            _KittiFile(    _raw_cal_vel2cam, parse_calib_vel2cam),
        # Raw Extract Dataset
        "raw_extract_cam_times":        _KittiFile(    _raw_ext_cam_tim, parse_raw_timestamp_file),
        "raw_extract_cam_data":         _KittiFile(    _raw_ext_cam_dat, parse_image),
        "raw_extract_oxts_times":       _KittiFile(    _raw_ext_oxt_tim, parse_raw_timestamp_file),
        "raw_extract_oxts_data":        _KittiFile(    _raw_ext_oxt_dat, parse_oxts),
        "raw_extract_velo_times":       _KittiFile(    _raw_ext_vel_tim, parse_raw_timestamp_file),
        "raw_extract_velo_times_start": _KittiFile(_raw_ext_vel_tim_sta, parse_raw_timestamp_file),
        "raw_extract_velo_times_end":   _KittiFile(_raw_ext_vel_tim_end, parse_raw_timestamp_file),
        "raw_extract_velo_data":        _KittiFile(    _raw_ext_vel_dat, parse_velo_txt),
        # Raw Synchronized Dataset
        "raw_sync_cam_times":           _KittiFile(    _raw_syn_cam_tim, parse_raw_timestamp_file),
        "raw_sync_cam_data":            _KittiFile(    _raw_syn_cam_dat, parse_image),
        "raw_sync_oxts_times":          _KittiFile(    _raw_syn_oxt_tim, parse_raw_timestamp_file),
        "raw_sync_oxts_data":           _KittiFile(    _raw_syn_oxt_dat, parse_oxts),
        "raw_sync_velo_times":          _KittiFile(    _raw_syn_vel_tim, parse_raw_timestamp_file),
        "raw_sync_velo_times_start":    _KittiFile(_raw_syn_vel_tim_sta, parse_raw_timestamp_file),
        "raw_sync_velo_times_end":      _KittiFile(_raw_syn_vel_tim_end, parse_raw_timestamp_file),
        "raw_sync_velo_data":           _KittiFile(    _raw_syn_vel_dat, parse_velo_bin),
    }

    # Parsers
    # ========================================================================================
    @classmethod
    def _load_file(cls, key, **kwargs):
        if key not in cls._kitti_file_dict:
            raise KittiError("Invalid file key {}. Valid values: {}".format(key, cls._kitti_file_dict.keys()))

        file_cfg = cls._kitti_file_dict[key]
        file_path = file_cfg.node.fpath(**kwargs)
        file_parser = file_cfg.parser

        return file_parser(file_path, **kwargs)

    @staticmethod
    def _update_kwargs(kwargs, new_kwargs, copy=True):
        if copy:
            tmp_kwargs = kwargs.copy()
        else:
            tmp_kwargs = kwargs
        tmp_kwargs.update(new_kwargs)
        return tmp_kwargs

    @classmethod
    def _load_odo_seq_file(cls, seq, key, **kwargs):
        tmp_kwargs = cls._update_kwargs(kwargs, {"seq": seq})
        return cls._load_file(key, **tmp_kwargs)

    @classmethod
    def load_odo_pose(cls, seq, **kwargs):
        return cls._load_odo_seq_file(seq, "odo_odo_pose", **kwargs)

    @classmethod
    def load_sem_pose(cls, seq, **kwargs):
        return cls._load_odo_seq_file(seq, "odo_sem_pose", **kwargs)

    @classmethod
    def load_odo_calib(cls, seq, **kwargs):
        return cls._load_odo_seq_file(seq, "odo_calib", **kwargs)

    @classmethod
    def load_odo_timestamps(cls, seq, **kwargs):
        return cls._load_odo_seq_file(seq, "odo_times", **kwargs)

    @classmethod
    def _load_odo_seq_frame_file(cls, seq, frame, key, **kwargs):
        tmp_kwargs = cls._update_kwargs(kwargs, {"seq": seq, "frame": frame})
        return cls._load_file(key, **tmp_kwargs)

    @classmethod
    def load_odo_velo(cls, seq, frame, **kwargs):
        return cls._load_odo_seq_frame_file(seq, frame, "odo_velo", **kwargs)

    @classmethod
    def load_odo_labels(cls, seq, frame, **kwargs):
        return cls._load_odo_seq_frame_file(seq, frame, "odo_labels", **kwargs)


    # String Formatting and Validation Methods
    # ========================================================================================
    @staticmethod
    def _validate_string(regex, string):
        """
        Check if a string matches a given regular expression.

        :param regex: Regular expression that the string must adhere to.
        :param string: String to be checked.
        :return: (bool) True if the string matches the regex, False otherwise.
        """

        return bool(regex.search(string))

    @staticmethod
    def _format_string(fmt, value):
        if isinstance(value, str):
            return value

        return fmt.format(value)

    @classmethod
    def format_kitti_str(cls, value, str_type):
        if str_type not in cls._str_format_dict:
            raise KeyError('Invalid Kitti string type ({}). Allowed ({}).'.format(str_type, cls._str_format_dict.keys()))

        str_cfg = cls._str_format_dict[str_type]

        # Format as string
        for t in str_cfg.types:
            if isinstance(value, t):
                value = cls._format_string(str_cfg.fmt, value)
                break
        else:
            raise TypeError("Invalid type ({}, {}) to format as ({}) string.".format(type(value), value, str_type))

        # Validate string
        if isinstance(value, str):
            # Compile REGEX if not already
            if str_cfg.re is None:
                str_cfg.re = re.compile(r"\A" + str_cfg.validation + r"\Z")

            if cls._validate_string(str_cfg.re, value):
                return value

        raise KittiError("Invalid KITTI {} ({}). Allowed values must match expression {}.".format(
            str_type, value, str_cfg.validation))

    @classmethod
    def format_odo_seq(cls, seq):
        return cls.format_kitti_str(seq, "seq")

    @classmethod
    def format_odo_velo(cls, frame):
        return cls.format_kitti_str(frame, "odo_velo")

    @classmethod
    def format_odo_labels(cls, frame):
        return cls.format_kitti_str(frame, "odo_labels")

    @classmethod
    def format_raw_drive_date(cls, date):
        return cls.format_kitti_str(date, "date")

    @classmethod
    def format_raw_drive(cls, drive):
        return cls.format_kitti_str(drive, "drive")

    @classmethod
    def format_raw_camera(cls, camera):
        return cls.format_kitti_str(camera, "raw_camera")

    @classmethod
    def format_raw_oxts(cls, frame):
        return cls.format_kitti_str(frame, "raw_oxts")

    @classmethod
    def format_raw_image(cls, frame):
        return cls.format_kitti_str(frame, "raw_image")

    @classmethod
    def format_raw_sync_velo(cls, frame):
        return cls.format_kitti_str(frame, "raws_velo")

    @classmethod
    def format_raw_extract_velo(cls, frame):
        return cls.format_kitti_str(frame, "rawe_velo")

    @classmethod
    def raw_kwargs(cls, req_args, kwargs):
        if "seq" in kwargs:
            if "date" in req_args and "date" not in kwargs:
                kwargs['date'] = cls.get_raw_seq_date(kwargs['seq'])
            if "drive" in req_args and "drive" not in kwargs:
                kwargs['drive'] = cls.get_raw_seq_date(kwargs['seq'])
        return kwargs

    # Directory Functions
    # ========================================================================================
    @classmethod
    def get_file_obj(cls, key):
        return cls._kitti_file_dict[key]

    @classmethod
    def get_file(cls, key, **kwargs):

        node = cls._kitti_file_dict[key].node
        kwargs = cls.raw_kwargs(node.args, kwargs)

        return node.fpath(**kwargs)

    @classmethod
    def set_root_dir(cls, root_dir):
        if not path.isdir(root_dir):
            raise OSError("Directory for KITTI dataset not found.\n{}".format(root_dir))

        cls._kitti_dir_structure.name = root_dir

    @classmethod
    def get_root_dir_obj(cls):
        return cls._kitti_dir_structure

    @classmethod
    def get_root_dir(cls, **kwargs):
        return cls.get_root_dir_obj().format(**kwargs)

    @classmethod
    def get_raw_seq_info(cls, seq):
        if seq not in cls._raw_seq_mapping:
            raise KittiError("Invalid sequence ({}). Valid values [{}].".format(seq, cls._raw_seq_mapping.keys()))

        return cls._raw_seq_mapping[seq]

    @classmethod
    def get_raw_seq_date(cls, seq):
        return cls.get_raw_seq_info(seq).date

    @classmethod
    def get_raw_seq_drive(cls, seq):
        return cls.get_raw_seq_info(seq).drive

    @classmethod
    def get_raw_seq_start_frame(cls, seq):
        return cls.get_raw_seq_info(seq).start_frame

    @classmethod
    def get_raw_seq_end_frame(cls, seq):
        return cls.get_raw_seq_info(seq).end_frame

    @classmethod
    def get_raw_seq_frame_range(cls, seq):
        return cls.get_raw_seq_start_frame(seq), cls.get_raw_seq_end_frame(seq)

    @classmethod
    def get_cameras(cls, mode=None, position=None):

        mode_opts = cls._raw_camera_cfg.keys()

        if mode is None:
            modes = mode_opts
        elif mode.upper() not in cls._raw_camera_cfg:
            raise KeyError("Invalid Camera Mode entered ({}). Valid options are ({}) or None for all modes.".format(
                mode, mode_opts))
        else:
            modes = [mode.upper()]

        ids = []
        for m in modes:
            camera_mode = cls._raw_camera_cfg[m]
            pos_opts = camera_mode.keys()
            positions = []

            if position is None:
                positions = pos_opts
            elif position.upper() in cls._raw_camera_cfg[m]:
                positions = [position.upper()]

            for p in positions:
                ids.append(camera_mode[p])

        return ids

    @classmethod
    def camera_is_grayscale(cls, camera):
        return camera in cls._raw_camera_cfg['GRAY'].items()

    @classmethod
    def camera_is_color(cls, camera):
        return camera in cls._raw_camera_cfg['COLOR'].items()

    @staticmethod
    def has_gt(seq):
        if seq < 0 or seq > 10:
            return False
        return True

    @classmethod
    def get_transforms(cls, calib):
        # Todo
        pass