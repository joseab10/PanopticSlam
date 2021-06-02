
# Third Party Libraries
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

# Project Libraries
import panoptic_slam.ros.utils as ru


class SemanticPointFilter:

    def __init__(self, dynamic_classes, class_field="class"):
        self._dynamic_classes = dynamic_classes
        self._class_field = class_field

        self._pcl_subscriber = rospy.Subscriber("points_labeled", PointCloud2, self._pcl_filter_callback)
        self._filtered_pcl_publisher = rospy.Publisher("points_filtered", PointCloud2, queue_size=5)

    def _pcl_filter_callback(self, msg):

        field_names = [f.name for f in msg.fields]

        if self._class_field not in field_names:
            return

        scan = ru.pcl2_msg_to_numpy(msg)
        filter_mask = np.logical_not(np.isin(scan[self._class_field], self._dynamic_classes))
        scan = scan[filter_mask]

        filtered_msg = ru.build_pcl2_msg(msg.header, msg.fields, scan, is_dense=True)
        self._filtered_pcl_publisher.publish(filtered_msg)
