"""Provides class for loading and parsing KITTI Odometry data"""

from os import path

import numpy as np

from panoptic_slam.kitti.data_loaders import KittiDataYielder
from panoptic_slam.kitti.exceptions import KittiGTError, KittiTimeError
import panoptic_slam.kitti.utils.utils as ku


class KittiOdomDataYielder(KittiDataYielder):

    def __init__(self, kitti_dir, seq, **kwargs):

        kitti_dir = ku.get_seq_dir(kitti_dir, seq)
        KittiDataYielder.__init__(kitti_dir, dataset_type="odom", **kwargs)

        self.seq = seq
        self._s_seq = ku.format_odo_seq(seq)

        self._dir_seq = ku.get_seq_dir(self.dir, seq)
        self._dir_label = path.join(self._dir_seq, "labels")

        self._timestamps = None
        self._ts_indexes = {}

        frame_step = kwargs.get("frame_step", None)
        self.frame_start = kwargs.get("start_frame", None)
        self.frame_end = kwargs.get("end_frame", None)
        self.frame_step = 1 if frame_step is None else frame_step
        self._is_frame_delimited = self.frame_start is not None and \
                                   self.frame_end is not None and \
                                   self.frame_step is not None

    def _in_frame_range(self, frame):
        if not self._is_frame_delimited:
            return True

        if self.frame_start is not None:
            if frame < self.frame_start:
                return False

        if self.frame_end is not None:
            if frame > self.frame_end:
                return False

        if self.frame_step is None:
            return True

        if self.frame_step == 1:
            return True

        return (frame - self.frame_start) % self.frame_step == 0

    def frame_range(self, max_frame=None):
        if max_frame is None:
            max_frame = 100000

        i = self.frame_start if self.frame_start is not None else 0
        step = self.frame_step if self.frame_step is not None else 1

        if self.frame_end is not None:
            end = self.frame_end
        else:
            if isinstance(max_frame, list):
                end = (len(max_frame) * step) + i
            elif isinstance(max_frame, int):
                end = max_frame
            else:
                raise TypeError("Invalid type for the max_frame parameter ({}).\
                                 Only int and list supported.".format(type(max_frame), max_frame))

        while i <= end:
            yield i
            i += step

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

        label_file = path.join(self._dir_label, ku.format_label(frame_index) + ".label")

        if not path.isfile(label_file):
            raise OSError("Requested label file not found.\n{}".format(label_file))

        label = np.fromfile(label_file, dtype=np.int32)
        label_class = np.bitwise_and(label, 0xFFFF)
        label_instance = np.right_shift(label, 16)

        return label_class, label_instance

    def yield_velodyne(self):
        velo_timestamps = self.get_timestamps()

        for t, i in zip(velo_timestamps, self.frame_range(velo_timestamps))
