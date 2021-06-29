from enum import Enum
from datetime import datetime

import numpy as np

import genpy
import rospy

import panoptic_slam.ros.utils as ru


class TimeMatching(Enum):
    Exact = 0
    Minimum = 1
    Previous = 2
    Next = 3

    @classmethod
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class KittiTimestamps:

    def __init__(self, timestamps, frame_indexes=None):

        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)

        if not isinstance(timestamps, np.ndarray):
            raise TypeError("Invalid type ({}) for timestamps.".format(type(timestamps)))

        if timestamps.dtype == np.object:
            if isinstance(timestamps[0], datetime):
                timestamps = np.array([ru.stamp_to_rospy(t).to_nsec() for t in timestamps])
            if isinstance(timestamps[0], rospy.Time):
                timestamps = np.array([t.to_nsecs() for t in timestamps])

        self.timestamps = timestamps

        ts_len = len(timestamps)

        if frame_indexes is None:
            frame_indexes = range(ts_len)

        f_len = len(frame_indexes)

        if f_len != ts_len:
            raise ValueError("Timestamps and Frame indexes have different sizes ({}, {}).".format(ts_len, f_len))

        # Build a lookup table for O(1) searches for exact timestamp matches.
        self._lookup_table = {t: i for t, i in zip(timestamps, frame_indexes)}

    def __repr__(self):
        return repr(self.timestamps)

    def __len__(self):
        return len(self.timestamps)

    def __array__(self):
        return self.timestamps

    def __getitem__(self, item):
        return self.timestamps[item]

    def get_frame_id(self, ts, ts_matching=TimeMatching.Minimum, max_error=None):
        """
        Get the frame id for a given timestamp

        :param ts: (int, rospy.Time or datetime.datetime) Timestamp to look for its frame index.
        :param ts_matching: (TimeMatching, Default: TimeMatching.Minimum) Timestamp matching method.
        :param max_error: (float, Default:None) Maximum allowable error between given timestamp and those in array.
        :return: (tuple) (frame_id, error)
        """
        if ts_matching not in TimeMatching:
            raise ValueError("Invalid TimeMatching method ({}).".format(ts_matching))

        if isinstance(ts, datetime):
            ts = ru.stamp_to_rospy(ts)

        if isinstance(ts, (rospy.Time, genpy.rostime.Time)):
            ts = ts.to_nsec()

        if ts in self._lookup_table:
            return self._lookup_table[ts], 0

        if ts_matching == TimeMatching.Exact:
            return None, None

        closest_ts_idx = 0

        if ts_matching == TimeMatching.Minimum:
            time_diff = self.timestamps - ts
            closest_ts_idx = np.argmin(np.abs(time_diff))

        if ts_matching == TimeMatching.Previous:
            potential_idx = np.where(self.timestamps <= ts)
            if len(potential_idx) == 0:
                return None, None

            closest_ts_idx = np.max(potential_idx)

        if ts_matching == TimeMatching.Next:
            potential_idx = np.where(self.timestamps >= ts)
            if len(potential_idx) == 0:
                return None, None

            closest_ts_idx = np.min(potential_idx)

        closest_ts = self.timestamps[closest_ts_idx]
        frame_id = self._lookup_table[closest_ts]
        err = ts - closest_ts

        if max_error is not None:
            if abs(err) > abs(max_error):
                return None, None

        return frame_id, err
