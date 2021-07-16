import numpy as np
from scipy.spatial import Delaunay


def grid_subsampling(points, voxel_size):
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
    # Returns the angle formed by sides a, b (i.e. opposite to c) in a triangle with sides a, b, c
    angle = np.square(a) + np.square(b) - np.square(c)
    angle = angle / (2 * a * b)
    angle = np.arccos(angle)

    return angle


def roll_array_cols(a, r):
    # https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    rows, cols = np.ogrid[:a.shape[0], :a.shape[1]]

    r[r < 0] += a.shape[1]
    col_idx = cols - r[:, np.newaxis]
    rolled_a = a[rows, col_idx]
    return rolled_a


def triangles_lengths_perimeters_areas(vertices):
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


def gaussian_likelihood(prop, value, sigma=1):
    likelihoods = prop - value
    likelihoods *= 1 / sigma
    likelihoods = - np.square(likelihoods)
    likelihoods = np.exp(likelihoods)

    if np.ndim(likelihoods) == 2:
        likelihoods = np.prod(likelihoods, axis=1)

    return likelihoods


def max_likelihoods(arr, n=3):
    a_max_likelihood = np.argpartition(arr, -n)[-n:]
    idx = a_max_likelihood[np.argsort(arr[a_max_likelihood])]

    return idx, arr[idx]


class Triangulation(Delaunay):

    def __init__(self, points3d, ignore_axis_for_triangulation=2, furthest_site=False, incremental=False, qhull_options=None):

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

        # self.angle_sigma = np.std(self.sorted_angles.ravel())
        self.angle_sigma = 5 * np.pi / 180  # 5 degrees
        # self.length_sigma = np.std(self.lengths.ravel())
        # self.length_sigma = 1000
        self.length_sigma = np.average(self.lengths.ravel()) * 0.1
        self.area_sigma = np.std(self.areas) / 1000
        self.perimeter_sigma = np.std(self.perimeters) / 1000

    def likelihoods_len(self, lengths):
        max_len_idx = - np.argmax(lengths)
        max_len_idx = max_len_idx + 3 if max_len_idx < 0 else max_len_idx
        sorted_len_idx = np.arange(3) - max_len_idx
        sorted_lengths = lengths[sorted_len_idx]
        return gaussian_likelihood(self.sorted_lengths, sorted_lengths, sigma=self.length_sigma)

    def find_triangle_by_lengths(self, lengths, n=3):
        return max_likelihoods(self.likelihoods_len(lengths), n=n)

    def save_vtk(self, file_path):
        with open(file_path, 'w') as f:

            f.write("# vtk DataFile Version 3.0\n")
            f.write("vtk output\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write("POINTS {} float\n".format(len(self.points3d)))
            np.savetxt(f, self.points3d, fmt="%.6f")
            #f.write(np.array2string(self.points3d.ravel(), precision=6))
            num_triangles = len(self.vertices)
            triangles = 3 * np.ones((num_triangles, 4))
            triangles[:, 1:] = self.vertices
            f.write("POLYGONS {} {}\n".format(num_triangles, 4 * num_triangles))
            np.savetxt(f, triangles, fmt="%d")
