import numpy as np
from scipy.spatial import Delaunay


def grid_subsampling(points, voxel_size):
    """
    Subsample a list of points according to a grid.

    :param points: (np.ndarray[nx3]) List of xyz points to be subsampled.
    :param voxel_size: (float) Size of the grid's voxels, so that only one point per voxel remains after subsampling.

    :return: (np.ndarray[mx3]) List of subsampled xyz points.
    """

    # https://colab.research.google.com/drive/1addhGqN3ZE1mIn4L6jQnnkVs7_y__qSE?usp=sharing#scrollTo=-g4CTM2knPYZ

    nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted = np.argsort(inverse)
    voxel_grid = {}
    grid_barycenter, grid_candidate_center = [], []
    last_seen = 0

    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        grid_barycenter.append(np.mean(voxel_grid[tuple(vox)], axis=0))
        grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] -
                                                                           np.mean(voxel_grid[tuple(vox)],
                                                                                   axis=0),
                                                                           axis=1).argmin()])
        last_seen += nb_pts_per_voxel[idx]

    return np.array(grid_candidate_center)


def cos_law(a, b, c):
    """
    Returns the angle formed by sides a, b (i.e. opposite to c) in a triangle with sides a, b, c.

    :param a: (float, np.ndarray[n]) Side length (or list of sides) A
    :param b: (float, np.ndarray[n]) Side length (or list of sides) B
    :param c: (float, np.ndarray[n]) Side length (or list of sides) C

    :return: (float,, np.ndarray[n]) Angle(s) formed between side(s) A and B
    """

    angle = np.square(a) + np.square(b) - np.square(c)
    angle = angle / (2 * a * b)
    angle = np.arccos(angle)

    return angle


def roll_array_cols(a, r):
    """
    Rolls an array a by r positions.

    :param a: (np.ndarray[nxm]) Array to be rolled
    :param r: (int) Number of positions to roll the array over its second axis.

    :return: (np.ndarray[nxm]) Array rolled r positions.
    """
    # https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    rows, cols = np.ogrid[:a.shape[0], :a.shape[1]]

    r[r < 0] += a.shape[1]
    col_idx = cols - r[:, np.newaxis]
    rolled_a = a[rows, col_idx]
    return rolled_a


def triangles_lengths_perimeters_areas(vertices):
    """
    Computes the angles, lengths, perimeters and areas of triangles defined by a list of vertices

    :param vertices: (np.ndarray[nx3x3]) Array of n triangles, each with 3 vertices in xyz.

    :return: (tuple):
                * (np.ndarray[nx3]) Array of the angles
                * (np.ndarrau[nx3]) Array of the lengths of the triangles' sides
                * (np.ndarray[n]) Array of the triangles' perimeters
                * (np.ndarray[n]) Array of the triangles' areas
    """
    # Lengths
    a = vertices[:, 0, :] - vertices[:, 1, :]
    b = vertices[:, 1, :] - vertices[:, 2, :]
    c = vertices[:, 2, :] - vertices[:, 0, :]
    a = np.sqrt(np.square(a).sum(1))
    b = np.sqrt(np.square(b).sum(1))
    c = np.sqrt(np.square(c).sum(1))

    angle_a = cos_law(b, c, a)
    angle_b = cos_law(a, c, b)
    angle_c = cos_law(a, b, c)

    angle_sum = angle_a + angle_b + angle_c

    angles = np.stack([angle_a, angle_b, angle_c]).T

    lengths = np.stack([a, b, c]).T

    # Perimeters:
    per = a + b + c

    # Areas:
    s = per / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    return angles, lengths, per, area


def gaussian_likelihood(prop, value, sigma=1.0):
    """
    Computes the likelihood of values according to a zero-mean gaussian distribution.

    :param prop: (np.ndarray[nxm]) List of properties to be considered as correct
    :param value: (np.ndarray[m]) List of values to compare.
    :param sigma: (float) Standard deviation of the normal distribution.

    :return: (np.ndarray[n]) List of likelihoods for the value w.r.t. to the property
    """

    likelihoods = prop - value
    likelihoods *= 1 / sigma
    likelihoods = - np.square(likelihoods)
    likelihoods = np.exp(likelihoods)

    if np.ndim(likelihoods) == 2:
        likelihoods = np.prod(likelihoods, axis=1)

    return likelihoods


def max_likelihoods(arr, n=3):
    """
    Gets the n items from arr with the greatest likelihoods

    :param arr: (np.ndarray[m]) Array from which to extract the n greatest likelihoods.
    :param n: (int, Default: 3) Number of greatest values to extract from array.

    :return: (np.ndarray[<=n], np.ndarray[<=n]) Array of indices of the maches, Sub-array of greatest values from arr
    """

    if len(arr) <= n:
        idx = np.argsort(arr)
        return idx, arr[idx]

    a_max_likelihood = np.argpartition(arr, -n)[-n:]
    idx = a_max_likelihood[np.argsort(arr[a_max_likelihood])]
    return idx, arr[idx]


class Triangulation(Delaunay):
    """
    Class for computing a 2D Delauney triangulation from a set of 3D points and subsequently allowing to search for
    a single noisy triangle.
    """

    def __init__(self, points3d, ignore_axis_for_triangulation=2, max_side_length=None, furthest_site=False, incremental=False, qhull_options=None):
        """
        Constructor

        :param points3d: (np.ndarray[nx3]) Array of 3D points from which to build a 2D triangulation.
        :param ignore_axis_for_triangulation: (int, Default: 2) Axis to ignore/project when doing the 2D triangulation.
                                              (0: x, 1: y, 2: z)
        :param max_side_length: (float, Default: None) Maximum length of any of the triangles' sides to be considered
                                valid. Invalid triangles are ignored during search and are not saved to a file.
                                If None, then all triangles are valid.
        :param furthest_site: Original Delaunay parameter.
        :param incremental: Original Delaunay parameter.
        :param qhull_options: Original Delaunay parameter.
        """
        self.points3d = points3d

        self.ignore_axis = ignore_axis_for_triangulation
        projected_axes = list(range(3))
        projected_axes.remove(self.ignore_axis)
        # Only triangulate x and y so as to not get Tetrahedral simplices
        Delaunay.__init__(self, points3d[:, projected_axes], furthest_site=furthest_site,
                          incremental=incremental, qhull_options=qhull_options)

        triangle_vertices = self.points3d[self.vertices]

        self.angles, self.lengths, self.perimeters, self.areas = triangles_lengths_perimeters_areas(triangle_vertices)

        # Roll them instead of sorting, so as to not alter the counterclockwise ordering.
        # Therefore, we don't allow triangles flipping, just rotating.
        # Thus, only the first length is guaranteed to be >= to the other two, but nothing can be said of the other two
        max_ang_idx = -np.argmax(self.angles, axis=1)
        self.sorted_angles = roll_array_cols(self.angles, max_ang_idx)
        max_len_idx = -np.argmax(self.lengths, axis=1)
        self.sorted_lengths = roll_array_cols(self.lengths, max_len_idx)

        self.max_side_length = max_side_length
        self.valid_triangles = None
        if self.max_side_length is not None:
            self.valid_triangles = (self.lengths.ravel() < self.max_side_length).reshape((-1, 3))
            self.valid_triangles = np.all(self.valid_triangles, axis=1)
            self.valid_triangles = np.argwhere(self.valid_triangles).ravel()

        # self.angle_sigma = np.std(self.sorted_angles.ravel())
        self.angle_sigma = 5 * np.pi / 180  # 5 degrees
        # self.length_sigma = np.std(self.lengths.ravel())
        # self.length_sigma = 1000
        if self.valid_triangles is not None:
            self.length_sigma = np.average(self.lengths[self.valid_triangles].ravel()) * 0.1
            self.area_sigma = np.std(self.areas[self.valid_triangles]) / 1000
            self.perimeter_sigma = np.std(self.perimeters[self.valid_triangles]) / 1000
        else:
            self.length_sigma = np.average(self.lengths.ravel()) * 0.1
            self.area_sigma = np.std(self.areas) / 1000
            self.perimeter_sigma = np.std(self.perimeters) / 1000

    def likelihoods_len(self, lengths):
        """
        Computes the likelihoods of the lengths of a given triangle, w.r.t. our own triangles.

        :param lengths: (np.ndarray[3]) Lengths of a triangle to compare with all of ours.

        :return: (np.ndarray[t]) Likelihood of the passed triangle of matching each of our triangles.
        """
        max_len_idx = - np.argmax(lengths)
        max_len_idx = max_len_idx + 3 if max_len_idx < 0 else max_len_idx
        sorted_len_idx = np.arange(3) - max_len_idx
        sorted_lengths = lengths[sorted_len_idx]

        if self.valid_triangles is not None:
            return gaussian_likelihood(self.sorted_lengths[self.valid_triangles], sorted_lengths,
                                       sigma=self.length_sigma)

        return gaussian_likelihood(self.sorted_lengths, sorted_lengths, sigma=self.length_sigma)

    def find_triangle_by_lengths(self, lengths, n=3):
        """
        Finds the n triangles from our triangulation that resemble the passed triangle the most by lengths.

        :param lengths: (np.ndarray[3]) Lengths defining a triangle to be searched in our triangulation.
        :param n: (int, Default: 3) Number of congruent triangle candidates to be returned.

        :return: (np.ndarray[<=n], np.ndarray[<=n]) Array of indices of the maches, Sub-array of greatest values from arr
        """

        return max_likelihoods(self.likelihoods_len(lengths), n=n)

    def save_vtk(self, file_path):
        """
        Saves the (valid) triangles from the triangulation as a *.vtk mesh file that can be viewed with pcl_viewer.

        :param file_path: (str) Path to the file to be saved.

        :return: None
        """

        with open(file_path, 'w') as f:

            f.write("# vtk DataFile Version 3.0\n"
                    "vtk output\n"
                    "ASCII\n"
                    "DATASET POLYDATA\n")
            f.write("POINTS {} float\n".format(len(self.points3d)))
            np.savetxt(f, self.points3d, fmt="%.6f")
            #f.write(np.array2string(self.points3d.ravel(), precision=6))
            num_triangles = len(self.valid_triangles) if self.valid_triangles is not None else len(self.vertices)
            triangles = 3 * np.ones((num_triangles, 4))
            triangles[:, 1:] = self.vertices[self.valid_triangles] if self.valid_triangles is not None else self.vertices
            f.write("POLYGONS {} {}\n".format(num_triangles, 4 * num_triangles))
            np.savetxt(f, triangles, fmt="%d")
