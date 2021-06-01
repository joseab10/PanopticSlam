import datetime as dt
from os import path

import numpy as np
import rospy

from panoptic_slam.kitti.utils.exceptions import KittiError
from panoptic_slam.kitti.utils import utils as ku


class KittiDataYielder:

    def __init__(self, kitti_dir, **kwargs):

        if not path.isdir(kitti_dir):
            raise OSError("Directory for KITTI dataset not found.\n{}".format(kitti_dir))

        self.dir = kitti_dir

        if "dataset_type" not in kwargs:
            raise KittiError("No dataset_type defined. Valid options: [raw, odom].")

        self._dataset_type = kwargs.get("dataset_type", None)
        self._parse_timestamp = None

        if self._dataset_type.lower() == "raw":
            self._parse_timestamp = ku.parse_raw_timestamp
            default_data_dirs = {
                'calib': "..",
                'oxts':  "oxts",
                'velo':  "velodyne_points",
                'cam0':  "image_00",
                'cam1':  "image_01",
                'cam2':  "image_02",
                'cam3':  "image_03",
            }
            self._timestamps_filename = "timestamps.txt"
        elif self._dataset_type.lower() == "odom":
            self._parse_timestamp = ku.parse_odom_timestamp
            default_data_dirs = {
                'calib':   ".",
                'oxts':    "oxts",
                'velo':    "velodyne",
                'labels':  "labels",
                # TODO: define the rest
                'cam0': "image_00",
                'cam1':  "image_01",
                'cam2':  "image_02",
                'cam3':  "image_03",
            }
            self._timestamps_filename = "times.txt"
        else:
            raise ValueError("Invalid dataset_type defined. Valid options: [raw, odom].")

        self._sub_dirs = {k: kwargs.get(k + "_dir""", d) for k, d in default_data_dirs.items()}

        frame_step = kwargs.get("frame_step", None)
        self.frame_start = kwargs.get("start_frame", None)
        self.frame_end = kwargs.get("end_frame", None)
        self.frame_step = 1 if frame_step is None else frame_step
        self._is_frame_delimited = \
            self.frame_start is not None and \
            self.frame_end is not None and \
            self.frame_step is not None

        self._time_offset_conf = kwargs.get("time_offset", None)
        self.time_offset = None

        self._loaded_timestamps = {}

        timestamp_override = kwargs.get("timestamp_override", None)
        self._timestamp_override = False
        if timestamp_override is not None:
            if isinstance(timestamp_override, (list, np.ndarray)):
                self._loaded_timestamps['override'] = timestamp_override
                self._timestamp_override = True
            else:
                raise TypeError("Invalid Timestamp Override type ({}, {}). Only a list of times is supported.")

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

    def _offset_timestamp(self, timestamp):
        if self._time_offset_conf is None:
            return timestamp

        if not self._time_offset_conf:
            return timestamp

        if self.time_offset is None:
            if isinstance(self._time_offset_conf, (float, dt.timedelta, dt.datetime, rospy.Time)):
                self.time_offset = self._time_offset_conf

            elif isinstance(self._time_offset_conf, str):
                if self._time_offset_conf.upper() == "FIRST":
                    self.time_offset = timestamp

                else:
                    raise ValueError("Invalid time offset configuration string ({}).".format(self._time_offset_conf))
            else:
                raise ValueError("Invalid time offset configuration ({}).".format(self._time_offset_conf))

        # Convert time offsets to timedelta objects
        # Consider floats as seconds
        if isinstance(self.time_offset, float):
            self.time_offset = dt.timedelta(seconds=self.time_offset)

        # Do the actual offsetting
        if isinstance(self.time_offset, (dt.timedelta, dt.datetime)):
            return timestamp - self.time_offset

        raise TypeError("Invalid Time Offset type ({}, {}).".format(type(self.time_offset), self.time_offset))

    def get_timestamps(self, data_key, force_load=False):
        if self._timestamp_override and not force_load:
            return self._loaded_timestamps['override']

        if self._dataset_type == "odom":
            # There is only one set of timestamps in Odometry Dataset, since it has been synchronized.
            data_key = "calib"

        if data_key in self._loaded_timestamps:
            return self._loaded_timestamps[data_key]

        timestamp_file = path.join(self.get_data_dir(data_key), self._timestamps_filename)

        if not path.isfile(timestamp_file):
            raise KittiError("Timestamp file ({}) not found.".format(timestamp_file))

        timestamps = []
        with open(timestamp_file, 'r') as f:
            for i, line in enumerate(f):
                if self._in_frame_range(i):
                    timestamps.append(self._offset_timestamp(self._parse_timestamp(line)))
                if self.frame_end is not None:
                    if i > self.frame_end:
                        break

        self._loaded_timestamps[data_key] = timestamps

        return timestamps

    def get_data_dir(self, key):
        if key not in self._sub_dirs:
            raise KittiError("Invalid directory key ({}). Valid values: [{}].".format(key, self._sub_dirs.keys()))

        return path.join(self.dir, self._sub_dirs[key])
