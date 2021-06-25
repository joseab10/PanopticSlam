import numpy as np
from os import path
import pypcd
from scipy.spatial.transform.rotation import Rotation

from panoptic_slam.geometry.algorithms.icp import icp, best_fit_transform
from panoptic_slam.kitti.data_loaders import KittiGTPosesLoader, TimeMatching
from panoptic_slam.panoptic_slam.metrics.pose_error_functions import pose_error_3d
from panoptic_slam.geometry.transforms.utils import transform_from_rot_trans
from panoptic_slam.geometry.point_cloud.utils import save_poses_as_pcd


def compute_pose_error(gt_poses, est_poses, **kwargs):
    """
    Computes the translational and rotational errors between Ground Truth and Estimated poses, after optionally aligning
    them using a matching algorithm.

    :param gt_poses: (np.ndarray[mx6]) Numpy array containing the Ground Truth positions.
                                       Each pose is defined as [x, y, z, roll, pitch, yaw]
    :param est_poses: (np.ndarray[mx6]) Numpy array containing the Estimated positions.
                                       Each pose is defined as [x, y, z, roll, pitch, yaw]
    :param kwargs: (dict) * match_alg (str, Default: "ls") Matching algorithm. Supported options are:
                                      - "ls" for Least-Squares. Use if pose correspondence between GT and EST is given.
                                      - "icp" for Iterative Closest Point. Use if correspondence is unknown.
                          * n_match_positions (int, Default: 10) Number of positions to use for matching and finding the
                                      relative transformation between GT and Est. None means to use all poses,
                                      while 0 disables the matching altogether.

    :return: (tuple) * tra_err (np.ndarray[m]) Euclidean distance between GT and (transformed) Estimated poses.
                     * rot_err (np.ndarray[m]) Rotational error (-pi, pi] between GT and (transformed) Estimated poses.
                     * tf (np.ndarray[4x4]) Best-fit homogeneous transformation matrix between GT and Estimated poses.
                     * t_est_poses (np.ndarray[mx6]) Transformed estimated poses according to tf.
    """

    if not isinstance(gt_poses, np.ndarray):
        raise TypeError("Invalid type for Ground Truth poses {}. Only Numpy array supported.".format(type(gt_poses)))
    if not isinstance(est_poses, np.ndarray):
        raise TypeError("Invalid type for Estimated poses {}. Only Numpy array supported.".format(type(est_poses)))

    n_match_positions = kwargs.get("n_match_positions", 10)

    match_alg = kwargs.get("match_alg", "ls").lower()

    gt_positions = gt_poses[:, :3]
    est_positions = est_poses[:, :3]
    tmp_est_poses = est_poses
    tf = np.eye(4)

    if n_match_positions is None:
        n_match_positions = gt_poses.shape[0]

    if n_match_positions > 0:
        n_match_positions = min(gt_poses.shape[0], n_match_positions)

        gt_match_pos = gt_positions[:n_match_positions]
        est_match_pos = est_positions[:n_match_positions]

        if match_alg == "icp":
            tf, _, _ = icp(est_match_pos, gt_match_pos)

        elif match_alg == "ls":
            tf = best_fit_transform(est_match_pos, gt_match_pos)

        else:
            raise ValueError("Invalid match algorithm ({}). Supported values: [icp, ls].".format(match_alg))

        # Construct tmp poses by reverting to homogeneous coordinates and applying relative transformation
        tmp_est_poses1 = np.zeros((est_poses.shape[0], 4, 4))
        tmp_est_poses1[:, :3, 3:] = est_positions.copy().reshape((-1, 3, 1))
        tmp_est_poses1[:, :3, :3] = Rotation.from_euler("xyz", est_poses[:, 3:]).as_dcm()
        tmp_est_poses1[:, -1, -1] = 1
        tmp_est_poses1 = np.matmul(tf, tmp_est_poses1)

        # Revert back to xyz,rpy vector
        tmp_est_poses = np.empty((est_poses.shape[0], 6))
        tmp_est_poses[:, :3] = tmp_est_poses1[:, :3, 3:].reshape((-1, 3))
        tmp_est_poses[:, 3:] = Rotation.from_dcm(tmp_est_poses1[:, :3, :3]).as_euler("xyz")

    tra_err, rot_err = pose_error_3d(gt_poses, tmp_est_poses)

    return tra_err, rot_err, tf, tmp_est_poses


def load_gt_poses(kitti_dir, seq):
    """
    Loads Ground Truth poses from KITTI dataset and formats them for easier comparison with LIO-SAM's estimated poses

    :param kitti_dir: (str) Path to the directory where the KITTI Data is stored. Parent of sequences/ and /raw.
    :param seq: (int) Number of the Ground Truth Sequence to use as baseline for the error computation.

    :return: (tuple) * gt_poses_loader (KittiGTPosesLoader) Loader object.
                     * gt_timestamps (KittiTimestamps) Ground Truth Timestamps object.
                     * gt_poses (np.ndarray[mx6]) Array of Ground Truth positions (xyz) and orientations (rpy).
    """

    gt_poses_loader = KittiGTPosesLoader(kitti_dir, seq, transform_to_velo_frame=False)
    gt_timestamps = gt_poses_loader.get_timestamps()
    gt_poses = gt_poses_loader.get_poses()
    gt_trans = gt_poses[:, :3, 3:].reshape((-1, 3))
    gt_rot = Rotation.from_dcm(gt_poses[:, :3, :3]).as_euler("xyz")
    gt_poses = np.hstack([gt_trans, gt_rot])

    return gt_poses_loader, gt_timestamps, gt_poses


def load_pcd_poses(pcd_file):
    """
    Loads Estimated Poses from a transformations.pcd file generated by LIO-SAM for easier comparison with KITTI's GT.

    :param pcd_file: (str) Path to the transformations.pcd Point Cloud file, generated by LIO-SAM.

    :return: (tuple) * pcd_timestamps (np.ndarray[m]) Array of timestamps in seconds.
                     * pcd_poses (np.ndarray[mx6]) Array of estimated positions (xyz) and orientations (rpy).
    """

    pcd_poses_pc = pypcd.point_cloud_from_path(pcd_file)
    pcd_poses_pc_data = pcd_poses_pc.pc_data
    pcd_timestamps = np.array(pcd_poses_pc_data['time'].tolist()) * 1e9
    pcd_poses = np.array(pcd_poses_pc_data[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].tolist())

    return pcd_timestamps, pcd_poses


def match_timestamps(gt_timestamps, est_timestamps, **kwargs):
    """
    Finds the correspondence between the Ground Truth and Estimated timestamps.

    :param gt_timestamps: (KittiTimestamps) KittiTimestamps object for the ground truth timestamps.
    :param est_timestamps: (list, np.ndarray[m]) Array of estimated timestamps in seconds.
    :param kwargs: (dict) * time_matching (TimeMatching, Default:Minimum) Time

    :return: (tuple): * gt_pose_indices (np.ndarray[n<=m]) Array containing the frame ids from the GT that matched with
                                        the estimated timestamps according to the time_matching criteria.
                      * gt_ts_match_err (np.ndarray[n<=m]) Array of errors in [ns] between matched GT and estimated
                                        timestamps.
    """

    time_matching = kwargs.get("time_matching", TimeMatching.Minimum)

    gt_pose_correspondence = np.array([gt_timestamps.get_frame_id(t, time_matching) for t in est_timestamps])
    gt_pose_indices = gt_pose_correspondence[:, 0].astype(int)
    gt_ts_match_err = gt_pose_correspondence[:, 1]

    return gt_pose_indices, gt_ts_match_err


def compute_pcd_pose_error(kitti_dir, seq, trans_pcd_file, **kwargs):
    """
    Computes the translational and rotational errors of a trajectory stored as PCD file by LIO-SAM
    in comparison to KITTI's Ground Truth poses.

    :param kitti_dir: (str) Path to the directory where the KITTI Data is stored. Parent of sequences/ and /raw.
    :param seq: (int) Number of the Ground Truth Sequence to use as baseline for the error computation.
    :param trans_pcd_file: (str) Path to the trajectory.pcd Point Cloud file, generated by LIO-SAM.
    :param kwargs: (dict) * save_path (str, Default: directory of pcd file) Directory in which to save output files.
                          * gt_poses_pcd_file (str, Default: <save_path>/gt_poses.pcd) Path to output GT poses file.
                          * trans_poses_pcd_file (str, Default: <save_path>/trans_est_poses.pcd) Path to transformed
                                                 estimated poses output file.
                          * pose_err_file (str, Default:<save_path>/pose_err.csv) Path to pose error csv file.
                          * ts_err_file (str, Default:<save_path/ts_err.csv) Path to timestamp error csv file.
                          * est_to_gt_tf_file (str, Default:<save_path>/est_to_gt_tf.csv) Path to csv file containing
                                                 the relative transformation matrix between GT and estimated poses.
                          * est_poses_file (str, Default:<save_path>/est_poses.txt) Path to estimated poses saved in
                                                 KITTI format for uploading for evaluation.
                          ** kwargs for compute_pose_error(), match_timestamps()

    :return: None
    """

    # Get Save paths for output files from kwargs
    save_path = kwargs.get("save_path", path.dirname(trans_pcd_file))
    gt_poses_pcd_file = kwargs.get("gt_poses_pcd_file", path.join(save_path, "gt_poses.pcd"))
    trans_poses_pcd_file = kwargs.get("trans_poses_pcd_file", path.join(save_path, "trans_est_poses.pcd"))
    pose_err_file = kwargs.get("pose_err_file", path.join(save_path, "pose_err.csv"))
    ts_err_file = kwargs.get("ts_err_file", path.join(save_path, "ts_err.csv"))
    est_to_gt_tf_file = kwargs.get("est_to_gt_tf_file", path.join(save_path, "est_to_gt_tf.csv"))
    est_poses_file = kwargs.get("est_poses_file", path.join(save_path, "est_poses.txt"))

    gt_poses_loader, gt_timestamps, gt_poses = load_gt_poses(kitti_dir, seq)
    est_timestamps, est_poses = load_pcd_poses(trans_pcd_file)
    gt_pose_indices, gt_ts_match_err = match_timestamps(gt_timestamps, est_timestamps, **kwargs)

    # Compute errors and relative transformation between estimated and gt poses
    tra_err, rot_err, tf, trans_est_poses = compute_pose_error(gt_poses[gt_pose_indices], est_poses, **kwargs)

    # Save output as files
    gt_poses_loader.save_as_pcd(gt_poses_pcd_file)
    print("Saved KITTI Ground Truth poses to PCD file: {}.".format(gt_poses_pcd_file))

    trans_est_positions = trans_est_poses[:, :3]
    trans_est_orientations = trans_est_poses[:, 3:]
    save_poses_as_pcd(trans_poses_pcd_file, trans_est_positions, trans_est_orientations,
                      gt_pose_indices, est_timestamps * 1e-9)
    print("Saved Estimated Poses (transformed to GT ref frame) to PCD file: {}.".format(trans_poses_pcd_file))

    pose_err = np.stack([gt_pose_indices, np.asarray(gt_timestamps)[gt_pose_indices] * 1e-9, tra_err, rot_err]).T
    np.savetxt(pose_err_file, pose_err, header="Frame_No. GT_Timestamp[s] Trans_Err_[m] Rot_Err_[rad]")
    print("Saved Translational and Rotational Pose Errors to csv file: {}.".format(pose_err_file))

    np.savetxt(ts_err_file, np.stack([gt_pose_indices, gt_ts_match_err]).T,
               header="GT_Frame_No Timestamp_error_(Est.t_-GT.t)[ns]")
    print("Saved Timestamp error CSV file: {}.".format(ts_err_file))

    np.savetxt(est_to_gt_tf_file, tf)
    print("Saved Homogeneous Transformation Matrix from Estimated to GT reference frame to: {}.".format(
        est_to_gt_tf_file))

    trans_rot_mat = Rotation.from_euler("xyz", trans_est_orientations).as_dcm()
    kitti_trans_tf = np.array([transform_from_rot_trans(r, t) for r, t in zip(trans_rot_mat, trans_est_positions)])
    kitti_trans_tf = kitti_trans_tf[:, :3, :]
    kitti_trans_tf = kitti_trans_tf.reshape((-1, 12))
    np.savetxt(est_poses_file, kitti_trans_tf)
    print("Saved Transformed Estimated Poses to KITTI-Formated TXT file: {}.".format(est_poses_file))


if __name__ == "__main__":
    compute_pcd_pose_error("/home/jose/Documents/Master_Thesis/dat/Kitti", 8, "/home/jose/Downloads/SEQ08_VANILLA/transformations.pcd")
