import numpy as np
import pypcd
from scipy.spatial.transform.rotation import Rotation as R

from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder, KittiRawDataYielder
import panoptic_slam.kitti.utils.utils as ku


class KittiGTPosesLoader:

    def __init__(self, kitti_dir, seq, **kwargs):

        self.kitti = KittiOdomDataYielder(kitti_dir, seq, **kwargs)
        date = ku.get_raw_seq_date(seq)
        drive = ku.get_raw_seq_drive(seq)
        start_frame = ku.get_raw_seq_start_frame(seq)
        end_frame = ku.get_raw_seq_end_frame(seq)
        kwargs['start_frame'] = start_frame
        kwargs['end_frame'] = end_frame
        self.raw_kitti = KittiRawDataYielder(kitti_dir, date, drive, **kwargs)
        self._odom_poses = kwargs.get("odom_poses", False)

        # Load Timestamps from Kitti Raw data and store them as ints in nanoseconds
        self._timestamps = self.raw_kitti.get_timestamps("velo")

        # Load poses from Kitti Odometry dataset (either the Semantic or Odometry poses, depending on odom_poses)
        poses = np.array(self.kitti.get_poses(self._odom_poses))

        # Load calibration and transform poses to velodyne reference frame for easier comparison with LIO-SAM poses
        cam1tovelo_tf = self.raw_kitti.get_transform("velo", "cam1")
        self._poses = np.matmul(cam1tovelo_tf, poses)

        # Extract positions and rotations as RPY angles
        self._positions = poses[:, :3, 3:].reshape(-1, 3)
        rot = poses[:, :3, :3]
        rot = R.from_dcm(rot)
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

        num_poses = len(self._timestamps)

        # Define PCL Fields
        pcl_type = np.dtype([("x", np.float32, 1), ("y", np.float32, 1), ("z", np.float32, 1),
                             ("intensity", np.float32, 1),
                             ("roll", np.float32, 1), ("pitch", np.float32, 1), ("yaw", np.float32, 1),
                             ("time", np.float64, 1)])

        # Build Point Cloud structure
        pcl_array = np.empty(num_poses, dtype=pcl_type)
        pcl_array['x'] = self._positions[:, 0]
        pcl_array['y'] = self._positions[:, 1]
        pcl_array['z'] = self._positions[:, 2]
        pcl_array['intensity'] = np.arange(num_poses)
        pcl_array['roll'] = self._orientations[:, 0]
        pcl_array['pitch'] = self._orientations[:, 1]
        pcl_array['yaw'] = self._orientations[:, 2]
        pcl_array['time'] = np.asarray(self._timestamps) * 1e-9  # Save timestamps as floats in seconds
        pcl = pypcd.PointCloud.from_array(pcl_array)

        pypcd.save_point_cloud(pcl, pcd_file)


if __name__ == "__main__":
    converter = KittiGTPosesLoader("/home/jose/Documents/Master_Thesis/dat/Kitti", 8)
    converter.save_as_pcd("/home/jose/Downloads/SEQ08_NAIVE/gt_trajectory.pcd")
