import numpy as np


def transform_from_rot_trans(r, t):
    """Transformation matrix from rotation matrix and translation vector."""

    r = r.reshape(3, 3)
    t = t.reshape(3, 1)

    return np.vstack((np.hstack([r, t]), [0, 0, 0, 1]))


def inv(transform_matrix):
    """Inverse of a rigid body transformation matrix"""

    r = transform_matrix[0:3, 0:3]
    t = transform_matrix[0:3, 3]
    t_inv = -1 * r.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = r.T
    transform_inv[0:3, 3] = t_inv

    return transform_inv


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def normalize_angles(angles):
    """
    Normalizes angles to range (-pi, pi]

    :param angles: (np.ndarray) Array of angles (in radians).
    :return: (np.ndarray) Array of angles normalized to (-pi, pi].
    """
    return np.where(angles > np.pi, angles - 2 * np.pi, angles)


def join_arrays(arrays):
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)

    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)

    dtype = sum((a.dtype.descr for a in arrays), [])

    return joint.ravel().view(dtype)