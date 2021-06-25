import numpy as np
from scipy.spatial.transform import Rotation as R

import panoptic_slam.geometry.transforms.utils as tu


def trans_error(ref_positions, est_positions):
    """
    Computes the translational euclidean error between two lists of n-dimensional vectors

    :param ref_positions: (np.ndarray[mxn]) Array of m reference position vectors, each of n-positional dimensions.
    :param est_positions: (np.ndarray[mxn]) Array of m estimated position vectors, each of n-positional dimensions.

    :return: (np.ndarray[m]) Array of m euclidean distances (translational errors).
    """

    if ref_positions.shape != est_positions.shape:
        raise ValueError("Reference ({}) and estimated ({}) positions must have the same shape.".format(
            ref_positions.shape, est_positions.shape))

    pose_diff = ref_positions - est_positions
    pose_err = np.sum(np.square(pose_diff), axis=1)
    pose_err = np.sqrt(pose_err)

    return pose_err


def rot_error_3d(ref_rot, est_rot, **kwargs):
    """
    Computes the rotational error between two 3D orientations by compounding one with the inverse of the other and
    finding the single rotation angle between the two.
    The rotation axis is ignored for the purpose of measuring the error.

    :param ref_rot: (np.ndarray[mx3]) Array of m reference orientations, each of 3 euler angles.
    :param est_rot: (np.ndarray[mx3]) Array of m estimated orientations, each of 3 euler angles.
    :param kwargs: (dict): * ref_axes (str, Default: "xyz"): Sequence of rotations for ref_rot.
                           * ref_degrees (bool, Default: False): Angles of ref_rot are in degrees, instead of radians.
                           * est_axes (str, Default: "xyz"): Sequence of rotations for est_rot.
                           * est_degrees (bool, Default: False): Angles of est_rot are in degrees, instead of radians.

    :return: (np.ndarray[m]) Array of m angular distances in (-pi, pi] (rotational errors).
    """

    # Parse kwargs
    ref_axes = kwargs.get("ref_axes", "xyz")
    ref_degrees = kwargs.get("ref_degrees", False)
    est_axes = kwargs.get("est_axes", "xyz")
    est_degrees = kwargs.get("est_degrees", False)

    # Convert the arrays of Roll, Pitch and Yaw angles to Rotation Objects
    rot_gt = R.from_euler(ref_axes, ref_rot, degrees=ref_degrees)
    rot_es = R.from_euler(est_axes, est_rot, degrees=est_degrees)

    # Compute the relative rotation as the composition of one of the rotations, and the inverse of the other,
    # represented as quaternions for convenience
    rel_rot = rot_gt * rot_es.inv()
    rel_rot = rel_rot.as_quat()

    # Get the w component of the quaternion, and extract the rotation angle from it
    w = rel_rot[:, 3]
    w = np.clip(w, -1, 1)
    angular_errors = 2 * np.arccos(w)
    angular_errors = tu.normalize_angles(angular_errors)

    return angular_errors


def pose_error_3d(gt_poses, estimated_poses):
    """
    Computes the translation and orientation errors between lists of estimated and ground truth poses.

    :param gt_poses: (np.ndarray[mx6]) List of ground truth poses, with columns (x, y, z, roll, pitch, yaw)
    :param estimated_poses: (np.ndarray[mx6]) List of estimated truth poses, with columns (x, y, z, roll, pitch, yaw)

    :return: (tuple): * tra_errors (np.ndarray[m]) List of translational errors between gt and estimated poses.
                      * rot_errors (np.ndarray[m]) List of rotational errors in (-pi, pi] between gt and est. poses.
    """

    tra_errors = trans_error(gt_poses[:, :3], estimated_poses[:, :3])
    rot_errors = rot_error_3d(gt_poses[:, 3:], estimated_poses[:, 3:])

    return tra_errors, rot_errors


def trajectory_lengths(trajectory):
    """
    Computes the travelled lengths between successive steps in a trajectory array.

    :param trajectory: (np.ndarray[mxn]) Array of dimensions representing m consecutive positions, each n-dimensional.

    :return: (np.ndarray[m-1]) Array of (m-1) euclidean distances between each position (i) and its preceding one (i-1).
    """
    xyz_lengths = trajectory[1:, :] - trajectory[:-1, :]
    return np.sqrt(np.dot(xyz_lengths, xyz_lengths))
