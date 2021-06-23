"""Provides several useful functions used for dealing with the KITTI Dataset"""

from collections import namedtuple
import datetime as dt
from os import path
import re

import cv2
import numpy as np

import panoptic_slam.kitti.utils.config as kc
from exceptions import KittiError
import panoptic_slam.geometry.transforms.utils as tu


def get_cameras(mode=None, position=None):

    mode_opts = kc.RAW_KITTI_CAMERA_CFG.keys()

    if mode is None:
        modes = mode_opts
    elif mode.upper() not in kc.RAW_KITTI_CAMERA_CFG:
        raise KeyError("Invalid Camera Mode entered ({}). Valid options are ({}) or None for all modes.".format(
            mode, mode_opts))
    else:
        modes = [mode.upper()]

    ids = []
    for m in modes:

        camera_mode = kc.RAW_KITTI_CAMERA_CFG[m]
        pos_opts = camera_mode.keys()
        positions = []

        if position is None:
            positions = pos_opts
        elif position.upper() in kc.RAW_KITTI_CAMERA_CFG[m]:
            positions = [position.upper()]

        for p in positions:
            ids.append(camera_mode[p])

    return ids


def camera_is_grayscale(camera):
    return camera in kc.RAW_KITTI_CAMERA_CFG['GRAY'].items()


def camera_is_color(camera):
    return camera in kc.RAW_KITTI_CAMERA_CFG['COLOR'].items()


def _validate_string(regex, string):
    return bool(regex.search(string))


def _format_string(fmt, value):
    if isinstance(value, dt.datetime):
        return value.strftime(fmt)

    if isinstance(value, str):
        return value

    return fmt.format(value)


def format_kitti_str(value, str_type):
    if str_type not in kc.KITTI_STR:
        raise KeyError('Invalid Kitti string type ({}). Allowed ({}).'.format(str_type, kc.KITTI_STR.keys()))

    str_cfg = dict(kc.KITTI_STR[str_type])

    # Format as string
    type_found = False
    for t in str_cfg['types']:
        if isinstance(value, t):
            value = _format_string(str_cfg['fmt'], value)
            type_found = True
            break
    if not type_found:
        raise TypeError("Invalid type ({}, {}) to format as ({}) string.".format(type(value), value, str_type))

    # Validate string
    if isinstance(value, str):
        # Compile REGEX if not already
        if 're' not in str_cfg:
            str_cfg['re'] = re.compile(r"\A" + str_cfg['valid'] + r"\Z")

        if _validate_string(str_cfg['re'], value):
            return value

    raise KittiError("Invalid KITTI {} ({}). Allowed values must match expression {}.".format(
        str_type, value, str_cfg['valid']))


def format_odo_seq(seq):
    return format_kitti_str(seq, "seq")


def format_odo_velo(frame):
    return format_kitti_str(frame, "odo velo")


def format_odo_velo_file(frame):
    return format_kitti_str(frame, "odo velo file")


def format_odo_labels(frame):
    return format_kitti_str(frame, "odo labels")


def format_odo_labels_file(frame):
    return format_kitti_str(frame, "odo labels file")


def format_raw_drive_date(date):
    return format_kitti_str(date, "date")


def format_raw_drive(drive):
    return format_kitti_str(drive, "drive")


def format_raw_camera(camera):
    return format_kitti_str(camera, "raw camera")


def format_raw_oxts(frame):
    return format_kitti_str(frame, "raw oxts")


def format_raw_oxts_file(frame):
    return format_kitti_str(frame, "raw oxts file")


def format_raw_image(frame):
    return format_kitti_str(frame, "raw image")


def format_raw_image_file(frame):
    return format_kitti_str(frame, "raw image file")


def format_raw_sync_velo(frame):
    return format_kitti_str(frame, "raws velo")


def format_raw_sync_velo_file(frame):
    return format_kitti_str(frame, "raws velo file")


def format_raw_extract_velo(frame):
    return format_kitti_str(frame, "rawe velo")


def format_raw_extract_velo_file(frame):
    return format_kitti_str(frame, "rawe velo file")


def format_poses_file(seq, odom=False):
    if not odom:
        return format_kitti_str(None, "poses sem file")

    return format_kitti_str(seq, "poses odo file")


# Parsers

def parse_odom_timestamp(timestamp_str):
    return dt.timedelta(seconds=float(timestamp_str))


def parse_raw_timestamp(timestamp_str):
    return dt.datetime.strptime(timestamp_str[:-4], '%Y-%m-%d %H:%M:%S.%f')


def has_gt(seq):
    if seq < 0 or seq > 10:
        return False

    return True


# Directory Functions

def get_seq_dir(kitti_dir, seq):

    seq = format_odo_seq(seq)

    kitti_seq_dir = path.join(kitti_dir, 'sequences', seq)

    if not path.isdir(kitti_seq_dir):
        raise OSError("Directory for KITTI sequence {} not found.\n{}".format(seq, kitti_seq_dir))

    return kitti_seq_dir


def get_raw_seq_info(seq, info):
    if seq not in kc.KITTI_RAW_SEQ_MAPPING:
        raise KittiError("Invalid sequence ({}). Valid values [{}].".format(seq, kc.KITTI_RAW_SEQ_MAPPING.keys()))

    if info not in kc.KITTI_RAW_SEQ_MAPPING[seq]:
        raise KeyError("Invalid Kitty Raw Mapping info ('{}'). Valid values {}.".format(
            info, kc.KITTI_RAW_SEQ_MAPPING[seq].keys()))

    return kc.KITTI_RAW_SEQ_MAPPING[seq][info]


def get_raw_seq_date(seq):
    return get_raw_seq_info(seq, 'date')


def get_raw_seq_drive(seq):
    return get_raw_seq_info(seq, 'drive')


def get_raw_seq_start_frame(seq):
    return get_raw_seq_info(seq, 'start_frame')


def get_raw_seq_end_frame(seq):
    return get_raw_seq_info(seq, 'end_frame')


def get_raw_seq_frame_range(seq):
    return get_raw_seq_start_frame(seq), get_raw_seq_end_frame(seq)


def get_raw_date_dir(kitti_dir, date):
    date = format_raw_drive_date(date)

    kitti_raw_date_dir = path.join(kitti_dir, 'raw', date)

    if not path.isdir(kitti_raw_date_dir):
        raise OSError("Directory for Raw KITTI date {} not found.\n{}".format(date, kitti_raw_date_dir))

    return kitti_raw_date_dir


def get_raw_drive_dir(kitti_dir, date, drive, sync=True):
    kitti_raw_date_dir = get_raw_date_dir(kitti_dir, date)

    drive = format_raw_drive(drive)
    date = format_raw_drive_date(date)

    sync_dir = "sync" if sync else "extract"
    drive_dir = date + "_drive_" + drive + "_" + sync_dir

    kitti_raw_drive_dir = path.join(kitti_raw_date_dir, drive_dir)

    if not path.isdir(kitti_raw_drive_dir):
        raise OSError("Directory for Raw KITTI drive {} not found.\n{}".format(drive, kitti_raw_drive_dir))

    return kitti_raw_drive_dir


def get_raw_seq_date_dir(kitti_dir, seq):
    date = get_raw_seq_date(seq)

    return get_raw_date_dir(kitti_dir, date)


def get_raw_seq_dir(kitti_dir, seq, sync=True):
    seq = format_odo_seq(seq)

    date = get_raw_seq_date(seq)
    drive = get_raw_seq_drive(seq)

    return get_raw_drive_dir(kitti_dir, date, drive, sync)


def parse_velo(scan):

    depth = np.linalg.norm(scan, 2, axis=1)
    pitch = np.arcsin(scan[:, 2] / depth)  # arcsin(z, depth)
    fov_down = -24.8 / 180.0 * np.pi
    fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
    proj_y = (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    proj_y *= 64  # in [0.0, H]
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(64 - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y = proj_y.reshape(-1, 1)
    scan = np.concatenate((scan, proj_y), axis=1)
    scan = list(scan.tolist())
    for i in range(len(scan)):
        scan[i][-1] = int(scan[i][-1])

    return scan


def parse_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    rx = tu.rotx(packet.roll)
    ry = tu.roty(packet.pitch)
    rz = tu.rotz(packet.yaw)
    r = rz.dot(ry.dot(rx))

    # Combine the translation and rotation into a homogeneous transform
    return r, t


def parse_oxts_packet(oxts_str):
    oxts = oxts_str.split()

    oxts[:-5] = [float(x) for x in oxts[:-5]]
    oxts[-5:] = [int(float(x)) for x in oxts[-5:]]

    return OxtsPacket(*oxts)


def parse_image(camera, img_path):
    cv_image = cv2.imread(img_path)

    if camera in kc.RAW_KITTI_CAMERA_CFG['GRAY'].values():
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    return cv_image


def stamp_to_sec(stamp):
    if isinstance(stamp, dt.datetime):
        return float(stamp.strftime("%s.%f"))

    if isinstance(stamp, dt.timedelta):
        return stamp.total_seconds()
