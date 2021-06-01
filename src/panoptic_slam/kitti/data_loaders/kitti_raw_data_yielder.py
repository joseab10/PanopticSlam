"""Provides class for loading and parsing KITTI Odometry data"""

from os import path

import numpy as np

from panoptic_slam.kitti.data_loaders import KittiDataYielder
from panoptic_slam.kitti.utils.exceptions import KittiError
import panoptic_slam.kitti.utils.utils as ku
import panoptic_slam.ros.transform_utils as tu


class KittiRawDataYielder(KittiDataYielder):

    def __init__(self, kitti_dir, date, drive, sync=True, **kwargs):

        self._kitti_dir = kitti_dir
        self.date = date
        self.drive = drive
        self.sync = sync

        data_dir = ku.get_raw_drive_dir(kitti_dir, date, drive, sync)
        KittiDataYielder.__init__(self, data_dir, dataset_type="raw", **kwargs)

        self._data_subdir = kwargs.get("data_subdir", "data")

        self._calib_functions = {
            # Calib Key: Calibration Function
            'cam2cam': self._parse_calib_cam2cam,
            'imu2vel': self._parse_calib_imu2vel,
            'vel2cam': self._parse_calib_vel2cam,
            'transforms': self._compute_transforms
        }

        self._calib = {}

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
            return tu.inv(transforms[inv_key])
        else:
            raise KittiError("No transform between ({}) and ({}) found.".format(from_frame, to_frame))

    def get_calib(self, calib_key):
        if calib_key not in self._calib:
            self._load_calib(calib_key)

        return self._calib[calib_key]

    def get_rigid_calib(self, calib_key):
        calib = self.get_calib(calib_key)
        return tu.transform_from_rot_trans(calib['R'], calib['T'])

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

            transform = tu.transform_from_rot_trans(rot, trans - origin)

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
