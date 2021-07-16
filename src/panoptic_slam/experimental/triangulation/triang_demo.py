from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from itertools import izip
from os import path, remove

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from triangulation import Triangulation, grid_subsampling
from panoptic_slam.geometry.transforms.utils import inv
from panoptic_slam.panoptic_slam.metrics.pose_error_functions import trajectory_lengths
from panoptic_slam.geometry.point_cloud.utils import save_poses_as_pcd
from panoptic_slam.io.utils import parse_path, mkdir
from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder, KittiRawDataYielder


class OrderedDefaultDict(OrderedDict, defaultdict):
    # https://stackoverflow.com/a/35968897
    def __init__(self, default_factory=None, *args, **kwargs):
        #in python3 you can omit the args to super
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory


if __name__ == "__main__":

    parser = ArgumentParser(description="Demo for triangulation loop-closure detection.")

    parser.add_argument("-d", "--kitti_dir", required=True, type=parse_path,
                        help="Path to the root of the KITTI dataset (Parent of the sequence/ and raw/ directories).")

    parser.add_argument("-o", "--output_dir", required=True, type=parse_path,
                        help="Path to the directory where the results will be saved.")

    parser.add_argument("-s", "--sequence", required=True, type=int,
                        help="Kitti sequence.")

    parser.add_argument("-f", "--class_filter", default=71, type=int,
                        help="Semantic class to remain after filtering to be used for the triangulation maps.")

    args = parser.parse_args()

    data_loader = KittiOdomDataYielder(args.kitti_dir, args.sequence)
    velo_to_cam_tf = np.eye(4)
    velo_to_cam_tf[:3, :] = data_loader.get_calib("Tr").reshape(3, 4)

    output_dir = args.output_dir
    mkdir(output_dir)

    scan_point_maps = []
    global_scan_point_last_indices = []
    global_triangle_maps = []
    loop_closures = []
    last_loop_closure_frame = 0
    positions = np.empty((0, 3))
    orientations = np.empty((0, 3, 3))
    traveled_distances = np.zeros(1)
    global_point_map = np.empty((0, 3))
    stats = OrderedDefaultDict(list)

    stats_file = path.join(output_dir, "stats.txt")
    if path.isfile(stats_file):
        remove(stats_file)
    first_stats_line = True

    min_loop_closure_distance = 50
    min_lookup_radius = 30

    print("Parsing KITTI Data and computing global maps.")
    for f, pose, scan, labels in tqdm(izip(data_loader.frame_range(), data_loader.get_poses(),
                                           data_loader.yield_velodyne(), data_loader.yield_labels(structured=False)),
                                      total=len(data_loader.get_poses())):

        t, scan = scan
        _, classes, instances = labels

        scan = scan[:, :3]  # Ignore intensity

        stats["frame"].append(f)
        stats["scan_points"].append(len(scan))

        # Filter scan to desired class
        filtered_indices = np.where(classes == args.class_filter)
        scan = scan[filtered_indices]
        instances = instances[filtered_indices]
        scan_instances = np.copy(instances)
        unique_instances = np.unique(instances)
        scan_unique_instances = np.copy(unique_instances)

        # Add un-transformed filtered scan points to scan array for later comparison.
        scan_point_maps.append(scan)
        position = pose[:3, 3:].T
        orientation = pose[:3, :3]
        if len(positions):
            frame_traveled_distance = position - positions[-1]
            frame_traveled_distance = np.array([np.sqrt(np.sum(np.square(frame_traveled_distance)))])
            traveled_distances = np.append(traveled_distances, frame_traveled_distance, axis=0)

        positions = np.append(positions, position, axis=0)
        orientations = np.append(orientations, orientation.reshape(1, 3, 3), axis=0)

        # if a scan does not provide any new information
        if len(scan) < 3:
            global_triangle_maps.append(global_triangle_maps[-1])
            continue

        stats["filtered_scan_points"].append(len(scan))
        scan_dbscan = DBSCAN()
        scan_instances = scan_dbscan.fit_predict(scan)
        scan_unique_instances = np.unique(scan_instances)

        scan_centroids = np.array([np.average(scan[scan_instances == i], axis=0) for i in scan_unique_instances if i >= 0])
        if len(scan_centroids) >= 3:
            scan_triangulation = Triangulation(scan_centroids, ignore_axis_for_triangulation=2)
            scan_triang_file = path.join(output_dir, "scan_triangulation_f{:04d}.vtk".format(f))
            scan_triangulation.save_vtk(scan_triang_file)
            scan_file = path.join(output_dir, "scan_f{:4d}.pcd".format(f))
            save_poses_as_pcd(scan_file, scan, frames=np.zeros(len(scan)))


        # Transform scan
        tf = np.matmul(pose, velo_to_cam_tf)
        homogeneous_scan = np.ones((scan.shape[0], 4))
        homogeneous_scan[:, :3] = scan
        homogeneous_scan = np.matmul(tf, homogeneous_scan.T).T
        transformed_scan = homogeneous_scan[:, :3]

        # Add points to global point map
        global_point_map = np.append(global_point_map, transformed_scan, axis=0)
        stats["global_map_points"].append(len(global_point_map))
        global_point_map = grid_subsampling(global_point_map, 0.1)
        stats["global_map_points_subsampled"].append(len(global_point_map))

        # Add length of accumulated global point maps up to this scan for later comparison
        global_scan_point_last_indices.append(len(global_point_map))

        # Cluster global map to get individual instances if class is not already segmented into instances
        #if len(unique_instances < 3):
        dbscan = DBSCAN()
        instances = dbscan.fit_predict(global_point_map)
        unique_instances = np.unique(instances)

        centroids = np.array([np.average(global_point_map[instances == i], axis=0) for i in unique_instances if i >= 0])

        stats["centroids"].append(len(centroids))

        triangulation = Triangulation(centroids, ignore_axis_for_triangulation=1)
        global_triangle_maps.append(triangulation)
        stats["triangles"].append(len(triangulation.vertices))

        global_points_file = path.join(args.output_dir, "global_map_f{:04d}.pcd".format(f))
        centroids_file = path.join(args.output_dir, "centroids_f{:04d}.vtk".format(f))

        save_poses_as_pcd(global_points_file, global_point_map, frames=np.zeros(len(global_point_map)))
        global_triangle_maps[-1].save_vtk(centroids_file)
        with open(stats_file, "a") as fh:
            if first_stats_line:
                fh.write(" ".join(stats.keys()) + "\n")
                first_stats_line = False

            row = " ".join([str(s[-1]) for s in stats.values()])
            fh.write(row + "\n")


        distance_since_last_loop_closure = np.sum(traveled_distances[last_loop_closure_frame:])
        retro_distance = np.cumsum(np.flip(traveled_distances[last_loop_closure_frame:]))
        latest_map_ckeck_idx = np.where(np.flip(retro_distance) > min_lookup_radius)

        if distance_since_last_loop_closure > min_loop_closure_distance and len(latest_map_ckeck_idx):
            loop_closure_map_idx = np.max(latest_map_ckeck_idx)
            if loop_closure_map_idx > last_loop_closure_frame:
                print("Loop Closure Search Condition fulfilled at frame {}, checking map up to {}.".format(f, loop_closure_map_idx))

                if len(scan_centroids) < 3:
                    print("No triangles found in scan. Loop-Closure detection stopped.")
                    continue



                # Find the triangle with the biggest area
                biggest_triangle_idx = np.argmax(scan_triangulation.areas)
                biggest_triangle_lengths = scan_triangulation.lengths[biggest_triangle_idx]
                triangle_matches, match_likelihoods = triangulation.find_triangle_by_lengths(biggest_triangle_lengths)

                if np.max(match_likelihoods) > 0.9:
                    print("Potential loop closure canditates: {} with likelihoods {}".format(triangle_matches, match_likelihoods))
                    last_loop_closure_frame = loop_closure_map_idx

