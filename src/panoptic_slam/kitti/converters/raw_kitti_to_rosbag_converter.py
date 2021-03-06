import tf
import numpy as np
from sensor_msgs.msg import PointField
from tqdm import tqdm

from panoptic_slam.kitti.converters.kitti_to_rosbag_converter import Kitti2RosBagConverter
from panoptic_slam.kitti.data_loaders import KittiRawDataYielder
import panoptic_slam.kitti.utils.utils as ku
import panoptic_slam.ros.utils as ru


class RawKitti2RosBagConverter(Kitti2RosBagConverter):

    def __init__(self, bag, kitti_dir, date, drive, sync=True, **kwargs):

        kitti_loader = KittiRawDataYielder(kitti_dir, date, drive, sync, **kwargs)
        Kitti2RosBagConverter.__init__(self, bag, kitti_loader, **kwargs)

        default_static_transforms = [
            (self.get_frame_id("base"), self.get_frame_id("imu"), self._tf_baselink_to_imu),
            (self.get_frame_id("imu"), self.get_frame_id("velo"), self.kitti_loader.get_transform("velo", "imu")),
            (self.get_frame_id("imu"), self.get_cam_frame_id(0),  self.kitti_loader.get_transform("cam0", "imu")),
            (self.get_frame_id("imu"), self.get_cam_frame_id(1),  self.kitti_loader.get_transform("cam1", "imu")),
            (self.get_frame_id("imu"), self.get_cam_frame_id(2),  self.kitti_loader.get_transform("cam2", "imu")),
            (self.get_frame_id("imu"), self.get_cam_frame_id(3),  self.kitti_loader.get_transform("cam3", "imu")),
        ]
        self._static_transforms = kwargs.get("static_transforms", default_static_transforms)

    def convert(self):
        if self._convert_static_tf:
            self.convert_static_tf()

        if self._convert_oxts:
            self.convert_oxts()

        if self._convert_cameras:
            self.convert_cameras()

        if self._convert_velodyne:
            self.convert_velodyne()

    def convert_static_tf(self):
        print("Exporting Static Transformations")
        transforms = []
        for tra in self._static_transforms:
            t = tra[2][0:3, 3]
            q = tf.transformations.quaternion_from_matrix(tra[2])
            tf_header = ru.build_header_msg(None, tra[0], seq=None)
            stamped_tf = ru.build_tf_stamped_transform(tf_header, tra[1], t, q)
            transforms.append(stamped_tf)

        for t in tqdm(self.kitti_loader.get_timestamps("oxts")):
            stamp = ru.stamp_to_rospy(t)
            for tra in transforms:
                tra.header.stamp = stamp

            tfm = ru.build_tf_msg(transforms)

            self.bag.write(self.get_msg_topic("tfs"), tfm, t=stamp)

    def _save_imu_data(self, header, stamp, topic, oxts):
        q = tf.transformations.quaternion_from_euler(oxts.roll, oxts.pitch, oxts.yaw)
        l_accel = [oxts.af, oxts.al, oxts.au]
        a_vel = [oxts.wf, oxts.wl, oxts.wu]
        imu_msg = ru.build_imu_msg(header, orientation=q, linear_acceleration=l_accel, angular_velocity=a_vel)
        self.bag.write(topic, imu_msg, t=stamp)

    def convert_oxts(self):

        if not self._convert_oxts:
            return

        tasks = ""
        tasks += " Raw IMU," if self._convert_raw_imu else ""
        tasks += " IMU," if self._convert_imu else ""
        tasks += " GPS Fix," if self._convert_gps_fix else ""
        tasks += " GPS Vel," if self._convert_gps_vel else ""
        tasks += " Dynamic TF," if self._convert_dynamic_tf else ""
        tasks = tasks[1:-1]

        print("Exporting OXTS ({}) Data:".format(tasks))

        oxts_generator = self.kitti_loader.yield_oxts()
        linear_ts = []

        if self._convert_raw_imu:
            oxts_timestamps = self.kitti_loader.get_timestamps("oxts")
            ts = [t for t in oxts_timestamps.as_sec()]
            x = np.asarray(range(len(ts)), dtype=np.float64)
            linear_ts = np.polyfit(x, ts, 1)

        for i, oxts_data in enumerate(tqdm(oxts_generator, total=len(self.kitti_loader.get_timestamps("oxts")))):
            t, oxts, transform = oxts_data
            stamp = ru.stamp_to_rospy(t)

            if self._convert_gps_fix:
                status = ru.build_navsatstatus_msg(None, 1)
                topic, frame_id = self.get_topic_and_frame("gps_fix")
                header = ru.build_header_msg(stamp, frame_id)
                gps_msg = ru.build_navsatfix_msg(header, oxts.lat, oxts.lon, oxts.alt, status)
                self.bag.write(topic, gps_msg, t=stamp)

            if self._convert_gps_vel:
                l_twist = [oxts.vf, oxts.vl, oxts.vu]
                a_twist = [oxts.wf, oxts.wl, oxts.wu]
                topic, frame_id = self.get_topic_and_frame("gps_vel")
                header = ru.build_header_msg(stamp, frame_id)
                twist_msg = ru.build_twiststamped_msg(header, linear_twist=l_twist, angular_twist=a_twist)
                self.bag.write(topic, twist_msg, t=stamp)

            if self._convert_imu:
                topic, frame_id = self.get_topic_and_frame("imu")
                header = ru.build_header_msg(t, frame_id)
                self._save_imu_data(header, stamp, topic, oxts)

            if self._convert_raw_imu:
                ts = linear_ts[0] * i + linear_ts[1]
                corrected_stamp = ru.stamp_to_rospy(ts)
                topic, frame_id = self.get_topic_and_frame("imu_raw")
                raw_imu_header = ru.build_header_msg(corrected_stamp, frame_id)
                self._save_imu_data(raw_imu_header, corrected_stamp, topic, oxts)
                imu_correct_topic, imu_correct_frame_id = self.get_topic_and_frame("imu_correct")
                correct_imu_header = ru.build_header_msg(corrected_stamp, imu_correct_frame_id)
                self._save_imu_data(correct_imu_header, corrected_stamp, imu_correct_topic, oxts)

            if self._convert_dynamic_tf:
                world_frame = self.get_frame_id("world")
                base_frame = self.get_frame_id("base")
                topic = self.get_msg_topic("tf")
                header = ru.build_header_msg(stamp, world_frame)
                t = transform[0:3, 3]
                q = tf.transformations.quaternion_from_matrix(transform)
                dyn_tf = ru.build_tf_stamped_transform(header, base_frame, t, q)
                tf_msg = ru.build_tf_msg([dyn_tf])
                self.bag.write(topic, tf_msg, stamp)

    def _save_camera_data(self, camera):
        camera_str = ku.format_raw_camera(camera)
        cam_key = self._get_camera_cfg_key(camera)
        topic, frame_id = self.get_topic_and_frame(cam_key)

        calib_header = ru.build_header_msg(None, frame_id)
        calib = self.kitti_loader.get_calib('cam2cam')
        w, h = tuple(calib["S_rect_{}".format(camera_str)].tolist())
        calib_dist_model = "plumb_bob"
        k = calib["K_{}".format(camera_str)]
        r = calib["R_rect_{}".format(camera_str)]
        d = calib["D_{}".format(camera_str)]
        p = calib["P_rect_{}".format(camera_str)]
        calib_msg = ru.build_camera_info_msg(calib_header, w, h, calib_dist_model, k, r, d, p)

        for t, image in tqdm(self.kitti_loader.yield_camera(camera)):
            stamp = ru.stamp_to_rospy(t)

            img_header = ru.build_header_msg(stamp, frame_id)
            encoding = "mono8" if ku.camera_is_grayscale(camera) else "bgr8"
            img_msg = ru.build_image_msg(img_header, image, encoding)
            calib_msg.header.stamp = stamp

            self.bag.write(topic + "/image_raw", img_msg, t=stamp)
            self.bag.write(topic + "/camera_info", calib_msg, t=stamp)

    def convert_cameras(self):
        print("Exporting Camera Data.")

        for c in self._camera_ids:
            print("Exporting Camera {}.".format(ku.format_raw_camera(c)))
            self._save_camera_data(c)

    def convert_velodyne(self):
        print("Exporting Velodyne Data.")
        topic, frame_id = self.get_topic_and_frame("velo")
        fields = [
            PointField("x",          0, PointField.FLOAT32, 1),
            PointField('y',          4, PointField.FLOAT32, 1),
            PointField('z',          8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('ring',      16, PointField.UINT16,  1)
        ]

        for t, scan in tqdm(self.kitti_loader.yield_velodyne(), total=len(self.kitti_loader.get_timestamps("velo"))):
            stamp = ru.stamp_to_rospy(t)
            parsed_scan = ku.parse_velo(scan)
            header = ru.build_header_msg(stamp, frame_id)
            velo_msg = ru.build_pcl2_msg(header, fields, parsed_scan, is_dense=True)
            self.bag.write(topic, velo_msg, t=stamp)
