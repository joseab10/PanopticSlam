import numpy as np
from scipy.spatial.transform.rotation import Rotation

from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder, KittiRawDataYielder
import panoptic_slam.kitti.utils.utils as ku
from panoptic_slam.geometry.point_cloud.utils import save_poses_as_pcd


class KittiGTPosesLoader:

    def __init__(self, kitti_dir, seq, **kwargs):

        self.kitti = KittiOdomDataYielder(kitti_dir, seq, **kwargs)
        date = ku.get_raw_seq_date(seq)
        drive = ku.get_raw_seq_drive(seq)
        start_frame = ku.get_raw_seq_start_frame(seq)
        end_frame = ku.get_raw_seq_end_frame(seq)
        kwargs['start_frame'] = start_frame
        kwargs['end_frame'] = end_frame
        transform_to_velo_frame = kwargs.get("transform_to_velo_frame", False)

        self.raw_kitti = KittiRawDataYielder(kitti_dir, date, drive, **kwargs)
        self._odom_poses = kwargs.get("odom_poses", False)

        # Load Timestamps from Kitti Raw data and store them as ints in nanoseconds
        self._timestamps = self.raw_kitti.get_timestamps("velo")

        # Load poses from Kitti Odometry dataset (either the Semantic or Odometry poses, depending on odom_poses)
        self._poses = np.array(self.kitti.get_poses(self._odom_poses))

        if transform_to_velo_frame:
            # Load calibration and transform poses to velodyne reference frame for easier comparison with LIO-SAM poses
            cam1tovelo_tf = self.raw_kitti.get_transform("velo", "cam1")
            self._poses = np.matmul(cam1tovelo_tf, self._poses)

        # Extract positions and rotations as RPY angles
        self._positions = self._poses[:, :3, 3:].reshape(-1, 3)
        rot = self._poses[:, :3, :3]
        rot = Rotation.from_dcm(rot)
        self._orientations = rot.as_euler("xyz", degrees=False)

    def get_timestamps(self):
        return self._timestamps

    def get_poses(self):
        return self._poses

    def get_positions(self):
        return self._positions

    def get_orientations(self):
        return self._orientations

    def save_as_pcd(self, pcd_file):
        save_poses_as_pcd(pcd_file, self._positions, self._orientations, None, self._timestamps.as_sec())
