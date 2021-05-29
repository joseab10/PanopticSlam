"""Provides class for loading and parsing KITTI Odometry data"""

from os import path

import numpy as np

from panoptic_slam.kitti.exceptions import KittiGTError, KittiTimeError
import panoptic_slam.kitti.utils as ku


class KittiDatasetYielder:

    def __init__(self, kitti_dir, seq, time_offset=None):
        if not path.isdir(kitti_dir):
            raise OSError("Directory for KITTI dataset not found.\n{}".format(kitti_dir))

        self.dir = kitti_dir

        self.seq = seq
        self._s_seq = ku.format_seq(seq)

        self._dir_seq = ku.get_seq_dir(self.dir, seq)
        self._dir_label = path.join(self._dir_seq, "labels")

        self._timestamps = None
        self._ts_indexes = {}
        self._time_offset = time_offset
        self.get_timestamps()

    def get_timestamps(self):
        if self._timestamps is not None:
            return self._timestamps

        timestamp_file = path.join(self._dir_seq, 'times.txt')

        timestamps = []
        with open(timestamp_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                t = ku.parse_timestamp(line)
                timestamps.append(t)
                t = t.total_seconds()
                self._ts_indexes[t] = i

        self._timestamps = np.array(timestamps)
        return self._timestamps

    def get_index_for_timestamp(self, timestamp, match="exact"):
        if not isinstance(timestamp, float):
            timestamp = timestamp.to_sec()

        if timestamp in self._ts_indexes:
            return self._ts_indexes[timestamp]

        if match == "exact":
            raise KittiTimeError("Timestamp {} has no exact frame index correspondence.".format(timestamp))

        idx = np.searchsorted(self._timestamps, timestamp, "left")

        if match == "last":
            if idx == 0:
                raise KittiTimeError("There are no frames before timestamp {}.".format(timestamp))
            return idx

        if match == "next":
            if idx >= len(self._timestamps):
                raise KittiTimeError("There are no frames after timestamp {}.".format(timestamp))
            return idx + 1

        if match == "closest":
            if idx == 0 or idx >= len(self._timestamps):
                return idx

            diff_arr = np.min(self._timestamps[idx - 1: idx + 1] - timestamp)
            idx = idx - 1 - np.argmin(diff_arr)
            return idx

        raise ValueError("Unknown matching option {}. Valid values are last | next | closest.".format(match))

    def get_labels_by_time(self, timestamp, match="exact"):

        idx = self.get_index_for_timestamp(timestamp, match=match)

        return self.get_labels_by_index(idx)

    def get_labels_by_index(self, frame_index):
        if not ku.has_gt(self.seq):
            raise KittiGTError("Sequence {} does not contain GT Semantic Labels.".format(self._s_seq))

        label_file = path.join(self._dir_label, ku.format_frame(frame_index) + ".label")

        if not path.isfile(label_file):
            raise OSError("Requested label file not found.\n{}".format(label_file))

        label = np.fromfile(label_file, dtype=np.int32)
        label_class = np.bitwise_and(label, 0xFFFF)
        label_instance = np.right_shift(label, 16)

        return label_class, label_instance
