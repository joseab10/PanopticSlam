"""Provides class for loading and parsing KITTI Odometry data"""

from os import path

import numpy as np

from panoptic_slam.kitti.data_loaders import KittiDataYielder
from panoptic_slam.kitti.utils.exceptions import KittiGTError
import panoptic_slam.kitti.utils.utils as ku


class KittiOdomDataYielder(KittiDataYielder):

    def __init__(self, kitti_dir, seq, **kwargs):

        kitti_dir = ku.get_seq_dir(kitti_dir, seq)
        KittiDataYielder.__init__(self, kitti_dir, dataset_type="odom", **kwargs)

        self.seq = seq
        self._s_seq = ku.format_odo_seq(seq)
        self._calib = None

    def get_labels_by_index(self, frame_index, structured=True):
        if not ku.has_gt(self.seq):
            raise KittiGTError("Sequence {} does not contain GT Semantic Labels.".format(self._s_seq))

        label_dir = self.get_data_dir("labels")
        label_file = ku.format_odo_labels_file(frame_index)

        label_file = path.join(label_dir, label_file)

        if not path.isfile(label_file):
            raise OSError("Requested label file not found.\n{}".format(label_file))

        label = np.fromfile(label_file, dtype=np.int32)
        label_class = np.bitwise_and(label, 0xFFFF)
        label_instance = np.right_shift(label, 16)

        if structured:
            label_class = label_class.astype(np.dtype([("class", np.int16, 1)]))
            label_instance = label_instance.astype(np.dtype([("instance", np.int16, 1)]))

        return label_class, label_instance

    def _load_calib(self):
        self._calib = ku.parse_calib_file(path.join(self.get_data_dir("calib"), "calib.txt"))

    def get_calib(self, key):
        if self._calib is None:
            self._load_calib()

        return self._calib[key]

    def get_velo_by_index(self, frame_index):
        velo_dir = self.get_data_dir("velo")

        filename = ku.format_odo_velo_file(frame_index)
        velo_file = path.join(velo_dir, filename)

        scan = np.fromfile(velo_file, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        return scan

    def yield_velodyne(self):
        velo_timestamps = self.get_timestamps(None)

        for t, i in zip(velo_timestamps, self.frame_range(velo_timestamps)):
            yield t, self.get_velo_by_index(i)

    def yield_labels(self, structured=True):
        velo_timestamps = self.get_timestamps(None)

        for t, i in zip(velo_timestamps, self.frame_range(velo_timestamps)):
            classes, instances = self.get_labels_by_index(i, structured=structured)
            yield t, classes, instances

    def get_poses(self, odom=False):

        if not ku.has_gt(self.seq):
            raise KittiGTError("KITTI Sequence {} has no ground truth poses.".format(self.seq))

        path_key = "pose_sem"
        if odom:
            path_key = "pose_odo"

        pose_dir = self.get_data_dir(path_key)
        pose_file = ku.format_poses_file(self.seq, odom=odom)

        pose_file = path.join(pose_dir, pose_file)
        poses = np.loadtxt(pose_file)
        # Add projection vector to complete (4x4) homogeneous transformation matrix
        poses = np.hstack([poses, np.zeros((len(poses), 3)), np.ones((len(poses), 1))])
        poses = poses.reshape((-1, 4, 4))

        return poses
