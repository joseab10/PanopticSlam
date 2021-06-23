# Standard Libraries
from collections import OrderedDict

# Third Party Libraries
import numpy as np
import rospy
from nav_msgs.msg import Path
from tf import transformations as trans

# Project Libraries
from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder, KittiRawDataYielder
import panoptic_slam.kitti.utils.utils as ku
from panoptic_slam.ros.utils import stamp_to_rospy
import panoptic_slam.geometry.transforms.utils as tu


class KittiGTPoseError:

    def __init__(self, kitti_dir, kitti_seq):

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
        self._gt_poses = self._kitti.get_poses()
        self._timestamps_dict = OrderedDict((stamp_to_rospy(t).to_nsec(), i) for i, t in enumerate(timestamps))
        self._timestamps_arr = np.array([stamp_to_rospy(t).to_nsec() for t in timestamps], dtype=np.int)
        self._poses = []
        self._ref_transforms = []
        self._exact_match = False
        self._ts_mismatch_ns = []

        self._path_subscriber = rospy.Subscriber("/lio_sam/mapping/path", Path, self._path_callback)

    def _path_callback(self, msg):

        empty_poses = [None] * len(self._gt_poses)
        ts_mismatch = [None] * len(self._gt_poses)

        for p in msg.poses:
            ts = p.header.stamp.to_nsec()

            if ts in self._timestamps_dict:
                frame = self._timestamps_dict[ts]
                ns_err = 0
            else:
                if self._exact_match:
                    continue

                frame = np.argmin(np.abs(self._timestamps_arr - ts))
                ns_err = self._timestamps_arr[frame] - ts

            pos, ori = p.pose.position, p.pose.orientation
            tra = np.array([pos.x, pos.y, pos.z])
            rot = np.array(trans.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])[:3, :3])
            empty_poses[frame] = tu.transform_from_rot_trans(rot, tra)
            ts_mismatch[frame] = ns_err

        self._poses.append(empty_poses)
        self._ts_mismatch_ns.append(ts_mismatch)
