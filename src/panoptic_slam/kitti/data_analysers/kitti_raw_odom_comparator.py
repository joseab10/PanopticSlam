from itertools import izip
import numpy as np

from panoptic_slam.kitti.data_loaders import KittiRawDataYielder, KittiOdomDataYielder
import panoptic_slam.kitti.utils.utils as ku


class KittiRawOdomComparator:
    def __init__(self, kitti_dir, seq, sync=True):
        self._kitti_dir = kitti_dir
        self._seq = seq
        self._sync = sync

        self._date = ku.get_raw_seq_date(self._seq)
        self._drive = ku.get_raw_seq_drive(self._seq)
        self._start_frame = ku.get_raw_seq_start_frame(self._seq)
        self._end_frame = ku.get_raw_seq_end_frame(self._seq)

        self._raw_data = KittiRawDataYielder(self._kitti_dir, self._date, self._drive, self._sync,
                                             start_frame=self._start_frame, end_frame=self._end_frame)
        self._odom_data = KittiOdomDataYielder(self._kitti_dir, self._seq)

    def compare_velo(self):

        raw_velo_ts = self._raw_data.get_timestamps('velo')
        odo_velo_ts = self._odom_data.get_timestamps(None)

        # Convert timestamps to time deltas with respect to first timestamp for easier comparison
        raw_velo_td = np.array([(t - raw_velo_ts[0]).total_seconds() for t in raw_velo_ts], dtype=np.float32)
        odo_velo_td = np.array([(t - odo_velo_ts[0]).total_seconds() for t in odo_velo_ts], dtype=np.float32)

        td = raw_velo_td - odo_velo_td
        td_non_zero = np.argwhere(td != 0)

        print("Velodyne data comparison between Raw and Odometry datasets")
        print("Raw Timestamps {},\tOdom Timestamps {}".format(len(raw_velo_ts), len(odo_velo_ts)))

        if len(td_non_zero) != 0:
            print("Found {} interval differences (w.r.t first timestamp) between raw and odometry datasets.".format(len(td_non_zero)))
        else:
            print("No interval differences (w.r.t. first timestamp) found between raw and odometry datasets found. Yay!!!")

        equal_scan_frames = []
        i = 0
        for raw_velo_data, odo_velo_data in izip(self._raw_data.yield_velodyne(), self._odom_data.yield_velodyne()):
            r_ts, r_velo = raw_velo_data
            o_ts, o_velo = odo_velo_data
            velo_diff = r_velo - o_velo
            diff_velo = np.argwhere(velo_diff != np.zeros(4))
            if len(diff_velo) != 0:
                print("Frame {:05d}. Found {} differences between Raw and Odometry scans.".format(i, len(diff_velo)))
            else:
                equal_scan_frames.append(i)
            i += 1

        print("No differences found in velodyne data in {}/{} frames.".format(len(equal_scan_frames), len(raw_velo_ts)))
