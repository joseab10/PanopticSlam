"""Provides class for loading and parsing KITTI Odometry data"""

import datetime as dt
from os import path

import numpy as np
import rospy

from panoptic_slam.kitti.exceptions import KittiError, KittiGTError, KittiTimeError
import panoptic_slam.kitti.utils as ku
import panoptic_slam.ros.utils as ru


class KittiRawDatasetYielder:

    def __init__(self, kitti_dir, date, drive, sync=True, **kwargs):
        if not path.isdir(kitti_dir):
            raise OSError("Directory for KITTI dataset not found.\n{}".format(kitti_dir))

        self.dir = kitti_dir

        self.date = date
        self.drive = drive
        self.sync = sync

        self._s_date = ku.format_drive_date(date)
        self._s_drive = ku.format_drive(drive)

        self.dir_drive = ku.get_raw_drive_dir(kitti_dir, date, drive, sync)

        default_data_dirs = {
            'calib': "..",
            'oxts':  "oxts",
            'velo':  "velodyne_points",
            'cam0':  "image_00",
            'cam1':  "image_01",
            'cam2':  "image_02",
            'cam3':  "image_03",
        }
        self._sub_dirs = {k: kwargs.get(k + "_dir""", d) for k, d in default_data_dirs.items()}
        self._data_subdir = kwargs.get("data_subdir", "data")

        self._calib_functions = {
            # Calib Key: Calibration Function
            'cam2cam': self._parse_calib_cam2cam,
            'imu2vel': self._parse_calib_imu2vel,
            'vel2cam': self._parse_calib_vel2cam,
            'transforms': self._compute_transforms
        }

        frame_step = kwargs.get("frame_step", None)

        self.frame_start = kwargs.get("start_frame", None)
        self.frame_end = kwargs.get("end_frame", None)
        self.frame_step = 1 if frame_step is None else frame_step
        self._is_frame_delimited = self.frame_start is not None and \
            self.frame_end is not None and \
            self.frame_step is not None

        self._time_offset_conf = kwargs.get("time_offset", None)
        self.time_offset = None

        self._calib = {}
        self._loaded_timestamps = {}

        timestamp_override = kwargs.get("timestamp_override", None)
        self._timestamp_override = False
        if timestamp_override is not None:
            if isinstance(timestamp_override, (list, np.ndarray)):
                self._loaded_timestamps['override'] = timestamp_override
                self._timestamp_override = True
            else:
                raise TypeError("Invalid Timestamp Override type ({}, {}). Only a list of times is supported.")

    def get_data_dir(self, key):
        if key not in self._sub_dirs:
            raise KittiError("Invalid directory key ({}). Valid values: [{}].".format(key, self._sub_dirs.keys()))

        return path.join(self.dir_drive, self._sub_dirs[key])

    def _load_calib(self, calib_key):
        if calib_key not in self._calib_functions:
            raise KeyError("Invalid calibration key ({}). Valid values [{}]".format(calib_key, self._calib.keys()))

        self._calib[calib_key] = self._calib_functions[calib_key]()

    def _parse_calib_file(self, calib_file):
        calib_file = path.join(self.get_data_dir("calib"), calib_file)
        return ku.parse_calib_file(calib_file)

    def _parse_calib_cam2cam(self):
        cam2cam_calib = self._parse_calib_file("calib_cam_to_cam.txt")
        data = {}

        tf_cam0unrect_velo = self.get_rigid_calib("vel2cam")

        # Load the rigid transformation from velodyne coordinates
        # to unrectified cam0 coordinates
        data['T_cam0_velo_unrect'] = tf_cam0unrect_velo

        # Create 3x4 projection matrices
        p_rect_00 = np.reshape(cam2cam_calib['P_rect_00'], (3, 4))
        p_rect_10 = np.reshape(cam2cam_calib['P_rect_01'], (3, 4))
        p_rect_20 = np.reshape(cam2cam_calib['P_rect_02'], (3, 4))
        p_rect_30 = np.reshape(cam2cam_calib['P_rect_03'], (3, 4))

        data['P_rect_00'] = p_rect_00
        data['P_rect_10'] = p_rect_10
        data['P_rect_20'] = p_rect_20
        data['P_rect_30'] = p_rect_30

        # Create 4x4 matrices from the rectifying rotation matrices
        r_rect_00 = np.eye(4)
        r_rect_00[0:3, 0:3] = np.reshape(cam2cam_calib['R_rect_00'], (3, 3))
        r_rect_10 = np.eye(4)
        r_rect_10[0:3, 0:3] = np.reshape(cam2cam_calib['R_rect_01'], (3, 3))
        r_rect_20 = np.eye(4)
        r_rect_20[0:3, 0:3] = np.reshape(cam2cam_calib['R_rect_02'], (3, 3))
        r_rect_30 = np.eye(4)
        r_rect_30[0:3, 0:3] = np.reshape(cam2cam_calib['R_rect_03'], (3, 3))

        data['T_rect_00'] = r_rect_00
        data['T_rect_10'] = r_rect_10
        data['T_rect_20'] = r_rect_20
        data['T_rect_30'] = r_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        t0 = np.eye(4)
        t0[0, 3] = p_rect_00[0, 3] / p_rect_00[0, 0]
        t1 = np.eye(4)
        t1[0, 3] = p_rect_10[0, 3] / p_rect_10[0, 0]
        t2 = np.eye(4)
        t2[0, 3] = p_rect_20[0, 3] / p_rect_20[0, 0]
        t3 = np.eye(4)
        t3[0, 3] = p_rect_30[0, 3] / p_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['TF_cam0_velo'] = t0.dot(r_rect_00.dot(tf_cam0unrect_velo))
        data['TF_cam1_velo'] = t1.dot(r_rect_00.dot(tf_cam0unrect_velo))
        data['TF_cam2_velo'] = t2.dot(r_rect_00.dot(tf_cam0unrect_velo))
        data['TF_cam3_velo'] = t3.dot(r_rect_00.dot(tf_cam0unrect_velo))

        # Compute the camera intrinsics
        data['K_cam0'] = p_rect_00[0:3, 0:3]
        data['K_cam1'] = p_rect_10[0:3, 0:3]
        data['K_cam2'] = p_rect_20[0:3, 0:3]
        data['K_cam3'] = p_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['TF_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['TF_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['TF_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['TF_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        return data

    def _parse_calib_imu2vel(self):
        return self._parse_calib_file("calib_imu_to_velo.txt")

    def _parse_calib_vel2cam(self):
        return self._parse_calib_file("calib_velo_to_cam.txt")

    def _compute_transforms(self):
        tf_data = {'TF_velo_imu': self.get_rigid_calib("imu2vel")}

        calib_cam_to_cam = self.get_calib("cam2cam")
        tf_data['TF_cam0_velo'] = calib_cam_to_cam['TF_cam0_velo']
        tf_data['TF_cam1_velo'] = calib_cam_to_cam['TF_cam1_velo']
        tf_data['TF_cam2_velo'] = calib_cam_to_cam['TF_cam2_velo']
        tf_data['TF_cam3_velo'] = calib_cam_to_cam['TF_cam3_velo']

        del calib_cam_to_cam['TF_cam0_velo']
        del calib_cam_to_cam['TF_cam1_velo']
        del calib_cam_to_cam['TF_cam2_velo']
        del calib_cam_to_cam['TF_cam3_velo']

        # Pre-compute the IMU to rectified camera coordinate transforms
        tf_data['TF_cam0_imu'] = tf_data['TF_cam0_velo'].dot(tf_data['TF_velo_imu'])
        tf_data['TF_cam1_imu'] = tf_data['TF_cam1_velo'].dot(tf_data['TF_velo_imu'])
        tf_data['TF_cam2_imu'] = tf_data['TF_cam2_velo'].dot(tf_data['TF_velo_imu'])
        tf_data['TF_cam3_imu'] = tf_data['TF_cam3_velo'].dot(tf_data['TF_velo_imu'])

        return tf_data

    def get_transform(self, from_frame, to_frame):
        transforms = self.get_calib("transforms")

        key = "TF_" + from_frame + "_" + to_frame
        inv_key = "TF_" + to_frame + "_" + from_frame

        if key in transforms:
            return transforms[key]
        elif inv_key in transforms:
            return ru.inv(transforms[inv_key])
        else:
            raise KittiError("No transform between ({}) and ({}) found.".format(from_frame, to_frame))

    def get_calib(self, calib_key):
        if calib_key not in self._calib:
            self._load_calib(calib_key)

        return self._calib[calib_key]

    def get_rigid_calib(self, calib_key):
        calib = self.get_calib(calib_key)
        return ku.transform_from_rot_trans(calib['R'], calib['T'])

    def _in_frame_range(self, frame):
        if not self._is_frame_delimited:
            return True

        if self.frame_start is not None:
            if frame < self.frame_start:
                return False

        if self.frame_end is not None:
            if frame > self.frame_end:
                return False

        if self.frame_step is None:
            return True

        if self.frame_step == 1:
            return True

        return (frame - self.frame_start) % self.frame_step == 0

    def frame_range(self, max_frame=None):
        if max_frame is None:
            max_frame = 100000

        i = self.frame_start if self.frame_start is not None else 0
        step = self.frame_step if self.frame_step is not None else 1

        if self.frame_end is not None:
            end = self.frame_end
        else:
            if isinstance(max_frame, list):
                end = (len(max_frame) * step) + i
            elif isinstance(max_frame, int):
                end = max_frame
            else:
                raise TypeError("Invalid type for the max_frame parameter ({}).\
                                 Only int and list supported.".format(type(max_frame), max_frame))

        while i <= end:
            yield i
            i += step

    def _offset_timestamp(self, timestamp):
        if self._time_offset_conf is None:
            return timestamp

        if not self._time_offset_conf:
            return timestamp

        if self.time_offset is None:
            if isinstance(self._time_offset_conf, (float, dt.timedelta, dt.datetime, rospy.Time)):
                self.time_offset = self._time_offset_conf

            elif isinstance(self._time_offset_conf, str):
                if self._time_offset_conf.upper() == "FIRST":
                    self.time_offset = timestamp

                else:
                    raise ValueError("Invalid time offset configuration string ({}).".format(self._time_offset_conf))
            else:
                raise ValueError("Invalid time offset configuration ({}).".format(self._time_offset_conf))

        # Convert time offsets to timedelta objects
        # Consider floats as seconds
        if isinstance(self.time_offset, float):
            self.time_offset = dt.timedelta(seconds=self.time_offset)

        # Do the actual offsetting
        if isinstance(self.time_offset, (dt.timedelta, dt.datetime)):
            return timestamp - self.time_offset

        raise TypeError("Invalid Time Offset type ({}, {}).".format(type(self.time_offset), self.time_offset))

    def get_timestamps(self, data_key, force_load=False):
        if self._timestamp_override and not force_load:
            return self._loaded_timestamps['override']

        if data_key in self._loaded_timestamps:
            return self._loaded_timestamps[data_key]

        timestamp_file = path.join(self.get_data_dir(data_key), "timestamps.txt")

        if not path.isfile(timestamp_file):
            raise KittiError("Timestamp file ({}) not found.".format(timestamp_file))

        timestamps = []
        with open(timestamp_file, 'r') as f:
            for i, line in enumerate(f):
                if self._in_frame_range(i):
                    timestamps.append(self._offset_timestamp(ku.parse_raw_timestamp(line)))
                if self.frame_end is not None:
                    if i > self.frame_end:
                        break

        self._loaded_timestamps[data_key] = timestamps

        return timestamps

    def set_timestamps(self, data_key, timestamps):
        if not isinstance(timestamps, list):
            raise TypeError("Invalid Timestamps ({}, {}). Must be a list of datetimes.")

        self._loaded_timestamps[data_key] = timestamps

    def yield_oxts(self):
        oxts_timestamps = self.get_timestamps("oxts")
        scale = None
        origin = None

        for t, i in zip(oxts_timestamps, self.frame_range(oxts_timestamps)):
            filename = ku.format_raw_oxts_file(i)
            oxts_file = path.join(self.get_data_dir("oxts"), self._data_subdir, filename)

            with open(oxts_file, 'r') as f:
                lines = f.readlines()

            if len(lines) != 1:
                raise KittiError("Invalid OXTS file ({}).".format(oxts_file))

            oxts = lines[0]

            oxts = ku.parse_oxts_packet(oxts)

            if scale is None:
                scale = np.cos(oxts.lat * np.pi / 180)

            rot, trans = ku.pose_from_oxts_packet(oxts, scale)

            if origin is None:
                origin = trans

            transform = ku.transform_from_rot_trans(rot, trans - origin)

            yield t, oxts, transform

    def yield_camera(self, camera):
        cam_key = "cam" + str(int(camera))
        cam_timestamps = self.get_timestamps(cam_key)

        for t, i in zip(cam_timestamps, self.frame_range(cam_timestamps)):
            filename = ku.format_raw_image_file(i)
            image_file = path.join(self.get_data_dir(cam_key), self._data_subdir, filename)

            cv_image = ku.parse_image(camera, image_file)

            yield t, cv_image

    def yield_velodyne(self):
        velo_timestamps = self.get_timestamps("velo")

        for t, i in zip(velo_timestamps, self.frame_range(velo_timestamps)):
            if self.sync:
                filename = ku.format_raw_sync_velo_file(i)
                velo_file = path.join(self.get_data_dir("velo"), self._data_subdir, filename)

                scan = np.fromfile(velo_file, dtype=np.float32)
                scan = scan.reshape((-1, 4))
            else:
                filename = ku.format_raw_extract_velo_file(i)
                velo_file = path.join(self.get_data_dir("velo"), self._data_subdir, filename)

                scan = np.loadtxt(velo_file, dtype=np.float32)

            yield t, scan
