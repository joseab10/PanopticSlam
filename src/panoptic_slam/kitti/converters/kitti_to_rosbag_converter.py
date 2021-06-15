import rosbag
import numpy as np

import panoptic_slam.kitti.utils.utils as ku


class Kitti2RosBagConverter:

    def __init__(self, bag, kitti_loader, **kwargs):

        if not isinstance(bag, rosbag.Bag):
            raise TypeError("Invalid ROSBag object. {} passed.".format(type(bag)))

        self.bag = bag

        self.kitti_loader = kitti_loader

        # Boolean Conversion Flags
        self._convert_static_tf = kwargs.get("convert_static_tf", False)
        self._convert_dynamic_tf = kwargs.get("convert_dynamic_tf", False)
        self._convert_imu = kwargs.get("convert_imu", False)
        self._convert_raw_imu = kwargs.get("convert_raw_imu", True)
        self._convert_gps_fix = kwargs.get("convert_gps_fix", True)
        self._convert_gps_vel = kwargs.get("convert_gps_vel", True)
        self._convert_cameras = kwargs.get("convert_cameras", False)
        self._convert_velodyne = kwargs.get("convert_velodyne", True)

        self._convert_oxts = self._convert_dynamic_tf or \
                             self._convert_imu or \
                             self._convert_raw_imu or \
                             self._convert_gps_fix or \
                             self._convert_gps_vel

        # Topics and Frame Configuration
        default_topics = {
            'tfs':         "/tf_static",
            'imu':         "/kiti/oxts/imu",
            'imu_raw':     "/imu_raw",
            'imu_correct': "/imu_correct",
            'gps_fix':     "/gps/fix",
            'gps_vel':     "/gps/vel",
            'velo':        "/points_raw"
        }

        default_frames = {
            'world':       "world",
            'map':         "map",
            'odom':        "odom",
            'base':        "base_link",
            'imu':         "imu_link",
            'imu_raw':     "imu_link",
            'imu_correct': "imu_enu_link",
            'gps_fix':     "imu_link",
            'gps_vel':     "imu_link",
            'velo':        "velodyne",
        }

        self._topics_cfg = {k: kwargs.get(k + "_topic", d) for k, d in default_topics.items()}
        self._frames_cfg = {k: kwargs.get(k + "_frame_id", d) for k, d in default_frames.items()}

        # Camera Topics and Frames
        default_camera_topics_frames = {
            0: {'frame_id': "camera_gray_left",   'topic': "/kitti/camera_gray_left"},
            1: {'frame_id': "camera_gray_right",  'topic': "/kitti/camera_gray_right"},
            2: {'frame_id': "camera_color_left",  'topic': "/kitti/camera_color_left"},
            3: {'frame_id': "camera_color_right", 'topic': "/kitti/camera_color_right"},
        }

        for k, v in default_camera_topics_frames.items():
            cam_key = self._get_camera_cfg_key(k)
            self._frames_cfg[cam_key] = kwargs.get(cam_key + "_frame_id", v['frame_id'])
            self._topics_cfg[cam_key] = kwargs.get(cam_key + "_topic", v['topic'])

        self._camera_modes = kwargs.get("camera_modes", None)
        self._camera_positions = kwargs.get("camera_positions", None)
        self._camera_ids = ku.get_cameras(self._camera_modes, self._camera_positions)

        # Static Transforms
        default_tf_baselink_to_imu = np.eye(4)
        default_tf_baselink_to_imu[0:3, 3] = [-2.71/2.0-0.05, 0.32, 0.93]
        self._tf_baselink_to_imu = kwargs.get("tf_baselink_to_imu", default_tf_baselink_to_imu)



    @staticmethod
    def _get_camera_cfg_key(camera):
        return "camera" + ku.format_raw_camera(camera)

    def get_frame_id(self, frame):
        return str(self._frames_cfg[frame])

    def get_cam_frame_id(self, camera):
        cid = self._get_camera_cfg_key(camera)
        return str(self._frames_cfg[cid])

    def get_msg_topic(self, msg):
        return str(self._topics_cfg[msg])

    def get_topic_and_frame(self, key):
        return self.get_msg_topic(key), self.get_frame_id(key)

    def get_cam_topic(self, camera):
        cid = self._get_camera_cfg_key(camera)
        return str(self._topics_cfg[cid])
