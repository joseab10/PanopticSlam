import numpy as np
import pypcd


def save_poses_as_pcd(pcd_file, positions, orientations=None, frames=None, timestamps=None):
    """
    Saves a set of (oriented and timestamped) points to a PCD file following the structure output by LIO-SAM
    for comparison purposes.

    :param pcd_file: (str) Path to the pcd file to be saved.
    :param positions: (np.ndarray[mx3], Default: None) List of x, y, z positions.
    :param orientations: (np.ndarray[mx3], Default: None) List of orientations as roll, pitch and yaw.
    :param frames: (np.ndarray[m], Default: None) List of frame numbers/ids. If none, consecutive numbers will be used.
    :param timestamps:  (np.ndarray[m]) Timestamps for the poses as floating point in seconds.

    :return: None
    """
    num_poses = len(positions)
    fields = [
        ("x", np.float32, 1), ("y", np.float32, 1), ("z", np.float32, 1),  # Position
        ("intensity", np.float32, 1)  # Intensity used for storing the frame number, just as in LIO-SAM
    ]

    if orientations is not None:
        if orientations.shape[0] != num_poses:
            raise ValueError("Number of orientations ({}) does not match number of poses ({}).".format(
                orientations.shape[0], num_poses))
        if orientations.shape[1] != 3:
            raise ValueError("Orientations should have 3 dimensions: roll, pitch and yaw.")
        fields.extend([("roll", np.float32, 1), ("pitch", np.float32, 1), ("yaw", np.float32, 1)])

    if frames is not None:
        if frames.shape[0] != num_poses:
            raise ValueError("Number of frames ({}) does not match number of poses ({}).".format(
                frames.shape[0], num_poses))
    else:
        frames = np.arange(num_poses)

    if timestamps is not None:
        if timestamps.shape[0] != num_poses:
            raise ValueError("Number of timestamps ({}) does not match number of poses ({}).".format(
                timestamps.shape[0], num_poses))
        fields.append(("time", np.float64, 1))

    # Define PCL Fields
    pcl_type = np.dtype(fields)

    # Build Point Cloud structure
    pcl_array = np.empty(num_poses, dtype=pcl_type)

    # Fill positions
    pcl_array['x'] = positions[:, 0]
    pcl_array['y'] = positions[:, 1]
    pcl_array['z'] = positions[:, 2]

    # Fill frame numbers
    pcl_array['intensity'] = frames

    # Fill orientations
    if orientations is not None:
        pcl_array['roll'] = orientations[:, 0]
        pcl_array['pitch'] = orientations[:, 1]
        pcl_array['yaw'] = orientations[:, 2]

    # Fill timestamps
    if timestamps is not None:
        pcl_array['time'] = timestamps

    pcl = pypcd.PointCloud.from_array(pcl_array)

    pypcd.save_point_cloud(pcl, pcd_file)
