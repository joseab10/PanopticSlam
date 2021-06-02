# Standard Libraries

# Third Party Libraries
import numpy as np
from numpy.lib import recfunctions as rfn
import rospy
from sensor_msgs.msg import PointCloud2, PointField

# Project Libraries
from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder, KittiRawDataYielder
from panoptic_slam.kitti.utils.exceptions import KittiError
import panoptic_slam.kitti.utils.utils as ku
from panoptic_slam.ros.utils import stamp_to_rospy, build_pcl2_msg
import panoptic_slam.ros.utils as ru


class KittiGTPanopticLabeler:

    def __init__(self, kitti_dir, kitti_seq):

        self._pcl_subscriber = rospy.Subscriber("points_raw", PointCloud2, self._pcl_callback)

        self._labeled_pcl_publisher = rospy.Publisher("points_labeled", PointCloud2, queue_size=5)

        self._kitti_dir = kitti_dir
        self._kitti_seq = kitti_seq

        self._kitti = KittiOdomDataYielder(self._kitti_dir, self._kitti_seq)

        date = ku.get_raw_seq_date(self._kitti_seq)
        drive = ku.get_raw_seq_drive(self._kitti_seq)
        start_frame = ku.get_raw_seq_start_frame(self._kitti_seq)
        end_frame = ku.get_raw_seq_end_frame(self._kitti_seq)

        self._raw_kitti = KittiRawDataYielder(self._kitti_dir, date, drive, sync=True,
                                              start_frame=start_frame, end_frame=end_frame)
        timestamps = self._raw_kitti.get_timestamps("velo")
        self._timestamp_table = {stamp_to_rospy(t): i for i, t in enumerate(timestamps)}

    def _pcl_callback(self, msg):

        ts = msg.header.stamp
        scan = ru.pcl2_msg_to_numpy(msg)
        point_len = scan.shape[0]

        # Get the offset as the offset of the last field + its length
        field_offset = msg.fields[-1].offset + ru.pcl2_field_len(msg.fields[-1]) * msg.fields[-1].count
        msg.fields.append(PointField("class", field_offset, PointField.UINT16, 1))
        msg.fields.append(PointField("instance", field_offset + 2, PointField.UINT16, 1))

        if ts in self._timestamp_table:
            try:
                frame_index = self._timestamp_table[ts]
                class_labels, instance_labels = self._kitti.get_labels_by_index(frame_index)

            except KittiError as e:
                class_labels = instance_labels = np.zeros(point_len)
                rospy.logwarn(e.message +
                              "Publishing 0 (unlabeled) class and instance labels for scan at time {}.".format(ts))
        else:
            class_labels = instance_labels = np.zeros()
            rospy.logwarn("No scan at time {}. Publishing 0 (unlabeled) class and instance labels.".format(ts))

        labels_type = np.dtype([("class", np.int16, 1), ("instance", np.int16, 1)])
        labels = np.zeros(point_len, dtype=labels_type)
        labels["class"] = class_labels
        labels['instance'] = instance_labels

        scan = rfn.merge_arrays((scan, labels), flatten=True)

        lbl_pcl_msg = build_pcl2_msg(msg.header, msg.fields, scan, is_dense=True)

        self._labeled_pcl_publisher.publish(lbl_pcl_msg)



