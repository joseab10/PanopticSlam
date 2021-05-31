import datetime as dt

from raw_kitti_to_rosbag_converter import RawKitti2RosBagConverter
from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder, KittiRawDataYielder
import panoptic_slam.kitti.utils.utils as ku


class RawKitti2LioSamSeqRosBagConverter:

    def __init__(self, bag, kitti_dir, seq, **kwargs):

        self.bag = bag
        self.kitti_dir = kitti_dir
        self.seq = seq

        # Get date, drive, start and end frame of Raw KITTI corresponding to Odometry's sequence
        self.date = dt.datetime.strptime(ku.get_raw_seq_date(self.seq), "%Y_%m_%d")
        self.drive = ku.get_raw_seq_drive(self.seq)
        self.start_frame = ku.get_raw_seq_start_frame(self.seq)
        self.end_frame = ku.get_raw_seq_end_frame(self.seq)

        self._crop_start = kwargs.get("crop_start", True)
        self._crop_end = kwargs.get("crop_end", True)

        if self._crop_start:
            kwargs['start_frame'] = self.start_frame
        if self._crop_end:
            kwargs['end_frame'] = self.end_frame

        # Copy kwargs for each converter (odom, sync, extract)
        odom_kwargs = {k: v for k, v in kwargs.items()}
        extract_kwargs = {k: v for k, v in kwargs.items()}
        sync_kwargs = {k: v for k, v in kwargs.items()}

        # Data split into sync and extract.
        # Messages in this dictionary will be taken from the extract dataset but not from the sync one,
        # whereas the remaining ones will be taken from the sync dataset but not from the extract one.
        default_conversions = {
            'raw_imu': True,
            'imu': False,
            'static_tf': False,
            'dynamic_tf': False,
            'gps_fix': True,
            'gps_vel': True,
            'velodyne': True,
            'cameras': False
        }
        default_odometry_stamped_msgs = []  # ['velodyne']
        default_extract_msgs = ['raw_imu']
        self._odometry_split = kwargs.get('odometry_split', default_odometry_stamped_msgs)
        self._extract_split = kwargs.get('extract_split', default_extract_msgs)

        if not set(self._odometry_split).isdisjoint(self._extract_split):
            intersection = set(self._odometry_split).intersection(self._extract_split)
            raise ValueError("Odometry split and Extract split have to be disjoint. \
                They both share these elements: ({}).".format(intersection))

        # Store user defined settings for converting raw extracted data, and set those to false
        for k, v in default_conversions.items():
            key = "convert_" + k
            user_arg = kwargs.get(key, v)

            if k in self._extract_split:
                odom_kwargs[key] = False
                sync_kwargs[key] = False
                extract_kwargs[key] = user_arg
            elif k in self._odometry_split:
                odom_kwargs[key] = user_arg
                sync_kwargs[key] = False
                extract_kwargs[key] = False
            else:
                odom_kwargs[key] = False
                sync_kwargs[key] = user_arg
                extract_kwargs[key] = False

        # Load the timestamps from the odometry dataset (NOT Raw), to use them for the velodyne scans
        self.odometry_loader = KittiOdomDataYielder(self.kitti_dir, self.seq)
        self._odometry_timestamps = self.odometry_loader.get_timestamps()
        odom_kwargs['timestamp_override'] = self._odometry_timestamps

        # Compute the time offset so that the first frame in the sequence for the velodyne data corresponds to time 0.0
        self.raw_loader = KittiRawDataYielder(self.kitti_dir, self.date, self.drive, sync=True)
        self._raw_velo_timestamps = self.raw_loader.get_timestamps("velo")
        # Base the time offset on the first frame of the velodyne data
        # dataset_start_time = self._raw_velo_timestamps[0]
        time_start = self._raw_velo_timestamps[self.start_frame]
        time_end = self._raw_velo_timestamps[self.end_frame]
        self._offset_time = kwargs.get("offset_time", False)
        if self._offset_time:
            time_offset = time_start
            sync_kwargs['time_offset'] = time_offset
            extract_kwargs['time_offset'] = time_offset

        # Compute the start and end frames for the IMU raw extract data
        start_extract_frame = None
        end_extract_frame = None

        if self._crop_start or self._crop_end:
            self.raw_extract_loader = KittiRawDataYielder(self.kitti_dir, self.date, self.drive, sync=False)
            self._raw_extract_oxts_timestamps = self.raw_extract_loader.get_timestamps("oxts")

            for i, t in enumerate(self._raw_extract_oxts_timestamps):
                if t >= time_start:
                    end_extract_frame = i
                    if start_extract_frame is None:
                        start_extract_frame = i
                if t > time_end:
                    break

        if self._crop_start:
            extract_kwargs['start_frame'] = start_extract_frame
        if self._crop_end:
            extract_kwargs['end_frame'] = end_extract_frame

        # Data converter from raw sync dataset using odometry timestamps (Mostly for velodyne data)
        self._odom_converter = RawKitti2RosBagConverter(self.bag, self.kitti_dir, self.date, self.drive, sync=True,
                                                        **odom_kwargs)
        # Data converter from raw sync dataset (For the rest of the messages)
        self._sync_converter = RawKitti2RosBagConverter(self.bag, self.kitti_dir, self.date, self.drive, sync=True,
                                                        **sync_kwargs)
        # Data converter from raw extract dataset (Mostly for the IMU Messages)
        self._extract_converter = RawKitti2RosBagConverter(self.bag, self.kitti_dir, self.date, self.drive, sync=False,
                                                           **extract_kwargs)

    def convert(self):
        print("\nConverting Raw Sync Data.")
        self._sync_converter.convert()
        print("\nConverting Raw Sync Data with stamps from Odometry dataset.")
        self._odom_converter.convert()
        print("\nConverting Raw Extract Data.")
        self._extract_converter.convert()

        return self.bag
