from enum import Enum
from datetime import datetime

import numpy as np

import genpy
import rospy

import panoptic_slam.ros.utils as ru
from exceptions import KittiTimeError


class KittiTimestamps:

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

    def __init__(self, ts_list_in_ns, frame_indexes=None):

        if isinstance(ts_list_in_ns, list):
            ts_list_in_ns = np.array(ts_list_in_ns)

        if not isinstance(ts_list_in_ns, np.ndarray):
            raise TypeError("Invalid type ({}) for timestamps.".format(type(ts_list_in_ns)))

        if ts_list_in_ns.dtype == np.object:
            if isinstance(ts_list_in_ns[0], datetime):
                ts_list_in_ns = np.array([ru.stamp_to_rospy(t).to_nsec() for t in ts_list_in_ns])
            if isinstance(ts_list_in_ns[0], rospy.Time):
                ts_list_in_ns = np.array([t.to_nsecs() for t in ts_list_in_ns])

        self.timestamps = ts_list_in_ns

        ts_len = len(ts_list_in_ns)

        if frame_indexes is None:
            frame_indexes = range(ts_len)

        f_len = len(frame_indexes)

        if f_len != ts_len:
            raise ValueError("Timestamps and Frame indexes have different sizes ({}, {}).".format(ts_len, f_len))

        # Build a lookup table for O(1) searches for exact timestamp matches.
        self._lookup_table = {t: i for t, i in zip(ts_list_in_ns, frame_indexes)}

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
        if ts_matching not in self.TimeMatching:
            raise ValueError("Invalid TimeMatching method ({}).".format(ts_matching))

        if isinstance(ts, datetime):
            ts = ru.stamp_to_rospy(ts)

        if isinstance(ts, (rospy.Time, genpy.rostime.Time)):
            ts = ts.to_nsec()

        if ts in self._lookup_table:
            return self._lookup_table[ts], 0

        if ts_matching == self.TimeMatching.Exact:
            raise KittiTimeError("No exact match found for stamp {}.".format(ts))

        closest_ts_idx = 0

        if ts_matching == self.TimeMatching.Minimum:
            time_diff = self.timestamps - ts
            closest_ts_idx = np.argmin(np.abs(time_diff))

        if ts_matching == self.TimeMatching.Previous:
            potential_idx = np.where(self.timestamps <= ts)
            if len(potential_idx) == 0:
                return None, None

            closest_ts_idx = np.max(potential_idx)

        if ts_matching == self.TimeMatching.Next:
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

    def as_nsec(self):
        """
        Returns the stored timestamps as a numpy array of integers in nanoseconds.

        :return: (np.ndarray) Array of timestamps as integers in nanoseconds.
        """

        return self.timestamps

    def as_sec(self):
        """
        Returns the stored timestamps as a numpy array of floats in seconds.

        :return: (np.ndarray) Array of timestamps as floats in seconds.
        """
        return self.timestamps.astype(np.float64) * 1e-9
