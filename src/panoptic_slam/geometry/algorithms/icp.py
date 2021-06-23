import numpy as np
from sklearn.neighbors import NearestNeighbors

import panoptic_slam.geometry.transforms.utils as tu


def assert_equal_shape(a, b):
    """
    Check if two numpy arrays have the same shape. If not, raise an exception
    :param a: (np.ndarray) Array A
    :param b: (np.ndarray) Array B

    :return: (None)
    """
    if a.shape != b.shape:
        raise ValueError("Shapes of source ({}) and destination ({}) arrays don't match.".format(
            a.shape, b.shape))


def best_fit_transform(src, dst):
    """
    Computes the best-fit transform that maps points from (n-dimensional) list A to B via centroid-translation and
    SVD rotation, explicitly solving least squares optimization.
    
    :param src: (np.ndarray[mxn]) Array of m source points, each of n-dimensions.
    :param dst: (np.ndarray[mxn]) Array of m destination points, each of n-dimensions.

    :return: (np.ndarray[(n+1)x(n+1)]) Homogeneous transformation matrix T that best transforms src into dst.
    """

    assert_equal_shape(src, dst)
    
    # Dimensionality of points
    n = src.shape[1]

    # Zero-center both of the point lists by subtracting their centroids (means)
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    src_zero_centered = src - centroid_src
    dst_zero_centered = dst - centroid_dst

    # Find the rotation matrix using Singular Value Decomposition
    approx_hessian = np.dot(src_zero_centered.T, dst_zero_centered)
    u, s, vt = np.linalg.svd(approx_hessian)
    rot = np.dot(vt.T, u.T)

    # Special reflection case
    if np.linalg.det(rot) < 0:
        vt[n - 1, :] *= -1
        rot = np.dot(vt.T, u.T)

    # Translation vector
    tra = centroid_dst.T - np.dot(rot, centroid_src.T)
    
    # Homogeneous transformation matrix
    tf = tu.transform_from_rot_trans(rot, tra)

    return tf


def nearest_neighbor(src, dst):
    """
    Find the nearest (euclidean) neighbour in dst for each point in src.
    
    :param src: (np.ndarray[mxn]) Array of m source points, each of n-dimensions.
    :param dst: (np.ndarray[mxn]) Array of m destination points, each of n-dimensions.
    :return: (tuple): * dist (np.ndarray[m]) List of (m) euclidean distances between each pair of point correspondences,
                      * indices (np.ndarray[m]) List of (m) indices of nearest point in dst for each point in src.
    """

    assert_equal_shape(src, dst)

    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(dst)
    distances, indices = neighbors.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()


def icp(src, dst, init_tf=None, max_iterations=20, tolerance=0.001):
    """
    Finds the best-fit transform that maps points in src to points in dst using the Iterative Closest Point algorithm,
    i.e., it iterates by finding point correspondence between nearest neighbors, then finding the best-fit transform,
    until either the error is below a certain tolerance, or a maximum number of iterations has been reached.
    
    :param src: (np.ndarray[mxn]) Array of m source points, each of n-dimensions.
    :param dst: (np.ndarray[mxn]) Array of m destination points, each of n-dimensions.
    :param init_tf: (np.ndarray[(n+1)x(n+1)], Default: None) Initial transformation matrix in homogeneous coordinates.
    :param max_iterations: (int, Default: 20) Maximum number of iterations to execute.
    :param tolerance: (float, Default: 0.001) ICP will stop if an error smaller than the tolerance is reached.

    :return: (tuple): * tf (np.ndarray[(n+1)x(n+1)]) Best-fit homogeneous transform matrix of src to dst,
                      * dist (np.ndarray[m]) List of (m) euclidean distances between each pair of point correspondences,
                      * i (int) Number of iterations that it took to converge.

    """

    assert_equal_shape(src, dst)

    # Dimensionality of points
    n = src.shape[1]

    # Add homogeneous coordinate to copy of point lists
    tmp_src = np.ones((src.shape[0], n + 1))
    tmp_src[:, :n] = np.copy(src)
    tmp_dst = np.ones((dst.shape[0], n + 1))
    tmp_dst[:, :n] = np.copy(dst)

    # If given, apply initial transformation
    if init_tf is not None:
        tmp_src = np.dot(init_tf, tmp_src)

    prev_error = 0
    i = 0
    distances = np.zeros(src.shape[0])

    for i in range(max_iterations):
        # Find point correspondence by nearest-neighbors
        distances, indices = nearest_neighbor(tmp_src[:, :n], tmp_dst[:, :n])

        # Compute the best-fit transformation between the current source and nearest destination points
        tf = best_fit_transform(tmp_src[:, :n], tmp_dst[indices, :n])

        # Update the current source points
        tmp_src = np.dot(tf, tmp_src)

        # Check error and break if below tolerance
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Compute the final transformation
    tf = best_fit_transform(src, tmp_src[:, :n])

    return tf, distances, i
