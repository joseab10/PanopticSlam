"""Provides class for loading and parsing KITTI Odometry data"""

from os import path

import numpy as np

from panoptic_slam.kitti.data_loaders import KittiDataYielder
from panoptic_slam.kitti.utils.exceptions import KittiGTError, KittiTimeError
import panoptic_slam.kitti.utils.utils as ku


class KittiOdomDataYielder(KittiDataYielder):

    def __init__(self, kitti_dir, seq, **kwargs):

        kitti_dir = ku.get_seq_dir(kitti_dir, seq)
        KittiDataYielder.__init__(self, kitti_dir, dataset_type="odom", **kwargs)

        self.seq = seq
        self._s_seq = ku.format_odo_seq(seq)

    def get_labels_by_index(self, frame_index):
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

        return label_class, label_instance

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
