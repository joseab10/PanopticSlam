import datetime as dt


from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
import numpy as np
import rosbag
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, NavSatFix, NavSatStatus
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointField
from tf2_msgs.msg import TFMessage


def stamp_to_rospy(stamp):
    if isinstance(stamp, rospy.Time):
        return stamp

    if isinstance(stamp, dt.datetime):
        stamp = float(stamp.strftime("%s.%f"))

    if isinstance(stamp, dt.timedelta):
        stamp = stamp.total_seconds()

    if isinstance(stamp, float):
        return rospy.Time.from_sec(stamp)

    raise TypeError("Invalid stamp data ({}, {}).".format(type(stamp), stamp))


def build_header_msg(stamp, frame_id, seq=None):
    header_msg = Header()

    if stamp is not None:
        header_msg.stamp = stamp_to_rospy(stamp)

    if frame_id is not None:
        header_msg.frame_id = frame_id

    if seq is not None:
        header_msg.seq = seq

    return header_msg


def build_imu_msg(header, orientation, linear_acceleration, angular_velocity):
    imu_msg = Imu()

    if header is not None:
        imu_msg.header = header

    if orientation is not None:
        imu_msg.orientation.x = orientation[0]
        imu_msg.orientation.y = orientation[1]
        imu_msg.orientation.z = orientation[2]
        imu_msg.orientation.w = orientation[3]

    if linear_acceleration is not None:
        imu_msg.linear_acceleration.x = linear_acceleration[0]
        imu_msg.linear_acceleration.y = linear_acceleration[1]
        imu_msg.linear_acceleration.z = linear_acceleration[2]

    if angular_velocity is not None:
        imu_msg.angular_velocity.x = angular_velocity[0]
        imu_msg.angular_velocity.y = angular_velocity[1]
        imu_msg.angular_velocity.z = angular_velocity[2]

    return imu_msg


def build_tf_stamped_transform(header, child_frame_id, translation, rotation_quaternion):
    transform = Transform()

    transform.translation.x = translation[0]
    transform.translation.y = translation[1]
    transform.translation.z = translation[2]

    transform.rotation.x = rotation_quaternion[0]
    transform.rotation.y = rotation_quaternion[1]
    transform.rotation.z = rotation_quaternion[2]
    transform.rotation.w = rotation_quaternion[3]

    stamped_transform = TransformStamped()

    stamped_transform.header = header
    stamped_transform.child_frame_id = child_frame_id
    stamped_transform.transform = transform

    return stamped_transform


def build_tf_msg(transform_list):
    tf_msg = TFMessage()

    for tf in transform_list:
        tf_msg.transforms.append(tf)

    return tf_msg


def build_camera_info_msg(header, width, height, distortion_model, k, r, d, p):
    camera_info_msg = CameraInfo()

    camera_info_msg.header = header
    if width is not None:
        camera_info_msg.width = width
    if height is not None:
        camera_info_msg.height = height
    if distortion_model is not None:
        camera_info_msg.distortion_model = distortion_model
    if k is not None:
        camera_info_msg.K = k
    if r is not None:
        camera_info_msg.R = r
    if d is not None:
        camera_info_msg.D = d
    if p is not None:
        camera_info_msg.P = p

    return camera_info_msg


def build_image_msg(header, cv2_image, encoding="passthrough"):
    bridge = CvBridge()
    img_msg = bridge.cv2_to_imgmsg(cv2_image, encoding=encoding)
    img_msg.header = header

    return img_msg


def build_navsatstatus_msg(status, service):
    stat_msg = NavSatStatus()
    if status is not None:
        stat_msg.status = status
    if service is not None:
        stat_msg.service = service

    return stat_msg


def build_navsatfix_msg(header, latitude, longitude, altitude, status):
    nav_msg = NavSatFix()
    if header is not None:
        nav_msg.header = header
    nav_msg.latitude = latitude
    nav_msg.longitude = longitude
    nav_msg.altitude = altitude
    if status is not None:
        nav_msg.status = status
    return nav_msg


def build_twiststamped_msg(header, linear_twist, angular_twist):
    twist_msg = TwistStamped()
    if header is not None:
        twist_msg.header = header

    if linear_twist is not None:
        twist_msg.twist.linear.x = linear_twist[0]
        twist_msg.twist.linear.y = linear_twist[1]
        twist_msg.twist.linear.z = linear_twist[2]

    if angular_twist is not None:
        twist_msg.twist.angular.x = angular_twist[0]
        twist_msg.twist.angular.y = angular_twist[1]
        twist_msg.twist.angular.z = angular_twist[2]

    return twist_msg


def build_pcl2_msg(header, fields, scan, is_dense=True):
    pcl2_msg = pcl2.create_cloud(header, fields, scan)
    pcl2_msg.is_dense = is_dense

    return pcl2_msg


_ROSBAG_COMPRESSION = {
    'none': rosbag.Compression.NONE,
    'bz2': rosbag.Compression.BZ2,
    'lz4': rosbag.Compression.LZ4
}


def parse_rosbag_compression(compression_str):
    k = compression_str.lower()
    if k in _ROSBAG_COMPRESSION:
        return _ROSBAG_COMPRESSION[k]

    print("WARNING: Invalid ROSBag compression algorithm. Using 'none'.")
    return _ROSBAG_COMPRESSION['none']


_PCL_FIELD_TYPES = {
    PointField.INT8:    {'len': 1, 'dtype': np.int8},
    PointField.UINT8:   {'len': 1, 'dtype': np.uint8},
    PointField.INT16:   {'len': 2, 'dtype': np.int16},
    PointField.UINT16:  {'len': 2, 'dtype': np.uint16},
    PointField.INT32:   {'len': 4, 'dtype': np.int32},
    PointField.UINT32:  {'len': 4, 'dtype': np.uint32},
    PointField.FLOAT32: {'len': 4, 'dtype': np.float32},
    PointField.FLOAT64: {'len': 8, 'dtype': np.float64},
}


def _pcl_field_info(pcl_field, info):
    if isinstance(pcl_field, PointField):
        pcl_field = pcl_field.datatype

    if pcl_field in _PCL_FIELD_TYPES:
        return _PCL_FIELD_TYPES[pcl_field][info]

    raise KeyError("Invalid PCL Fields Type ({}).".format(pcl_field))


def pcl2_field_len(pcl_field):
    return _pcl_field_info(pcl_field, "len")


def pcl2_field_type_to_np_dtype(pcl_field, str_rep=False):
    data_type = _pcl_field_info(pcl_field, "dtype")

    if str_rep:
        data_type = np.dtype(data_type).str

    if isinstance(pcl_field, PointField):
        return pcl_field.name, data_type, pcl_field.count

    return data_type


def pcl2_msg_to_numpy(pcl2_msg):
    pcl_field_types = [pcl2_field_type_to_np_dtype(f, str_rep=True) for f in pcl2_msg.fields]
    return np.frombuffer(pcl2_msg.data, dtype=pcl_field_types)

