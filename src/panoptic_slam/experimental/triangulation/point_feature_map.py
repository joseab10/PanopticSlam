import numpy as np
from sklearn.cluster import DBSCAN


class PointFeatureMap:

    def __init__(self, clustering_method=None, subsampling_method=None):
        """
        Creates, maintains and updates a point-feature map built from point clouds.

        :param clustering_method: (callable/None) Method for clustering points together. If None, then DBScan is used.
        :param subsampling_method: (callable/None) Method for sub-sampling points. If None, Grid Sub-sampling is used.
        """

        if not (clustering_method is None or callable(clustering_method)):
            raise TypeError("Invalid clustering method. Only callables or None supported.")

        self.clustering_method = clustering_method

        if not (subsampling_method is None or callable(subsampling_method)):
            raise TypeError("Invalid subsampling method. Only callables or None supported.")
        self.subsampling_method = subsampling_method

        self.features = np.empty((0, 3))
        self.feature_point_indices = np.empty((0, 2), dtype=int)
        self.bounding_boxes = np.empty((0, 2, 3))  # Feature, min/max, xyz
        self.points = np.empty((0, 3))

    def _append_object_points(self, points, centroid=None, bb=None):
        """
        Appends the points (and properties) of an object to the Feature map

        :param points: (np.ndarray[nx3]) Array of xyz points
        :param centroid: (np.ndarray[3], Default: None) Position of the point's centroid.
                         If not given, it will be computed.
        :param bb: (np.ndarray[2x3], Default: None) Position of the bounding box min and max
                         corners ((x,y,z)_min, (x,y,z)_max). If not given, it will be computed.

        :return: None
        """

        if centroid is None:
            centroid = np.average(points, axis=0)

        if bb is None:
            bb_min = np.min(points, axis=0)
            bb_max = np.max(points, axis=0)
            bb = np.vstack([bb_min, bb_max])

        self.features = np.append(self.features, centroid.reshape((1, 3)), axis=0)
        self.bounding_boxes = np.append(self.bounding_boxes, bb.reshape((1, 2, 3)), axis=0)

        if self.subsampling_method is not None:
            points = self.subsampling_method(points)

        last_idx = len(self.points)
        self.feature_point_indices = np.append(self.feature_point_indices,
                                               np.array([[last_idx, last_idx + len(points)]]), axis=0)
        self.points = np.append(self.points, points.reshape((-1, 3)), axis=0)

    def add_object_points(self, points):
        """
        Adds an object's points (and other properties) to the map, by first checking if the object intersects with
        with other objects already in the map. If so, then it does a reclustering and resampling.

        :param points: (np.ndarray[nx3]) Array of n points in xyz to be added.

        :return: None
        """

        centroid = np.average(points, axis=0)
        bb_min = np.min(points, axis=0)
        bb_max = np.max(points, axis=0)
        bb = np.vstack([bb_min, bb_max])

        # Check if object intersects other objects already in the map
        int_cond1 = np.greater_equal(bb_max, self.bounding_boxes[:, 0, :])
        int_cond2 = np.less_equal(bb_min, self.bounding_boxes[:, 1, :])
        int_cond = np.logical_and(int_cond1, int_cond2)
        int_cond = np.all(int_cond, axis=1)
        int_idx = np.nonzero(int_cond)[0]

        # If new object does not intersect, just add it
        if len(int_idx) == 0:
            self._append_object_points(points, centroid, bb)
            return

        # Otherwise, we are in some trouble...
        # We have to remove the old objects
        remaining_object_indices = np.setdiff1d(np.arange(len(self.features)), int_idx)

        self.bounding_boxes = self.bounding_boxes[remaining_object_indices]
        self.features = self.features[remaining_object_indices]

        int_points = np.empty((0, 3))
        for i in int_idx:
            int_point_indices = self.feature_point_indices[i]
            start = int_point_indices[0]
            end = int_point_indices[1]
            step_int_points = self.points[start:end, :]
            int_points = np.append(int_points, step_int_points, axis=0)
            self.points = np.delete(self.points, np.arange(start, end), axis=0)
            self.feature_point_indices[i + 1:, :] -= len(step_int_points)

        self.feature_point_indices = self.feature_point_indices[remaining_object_indices]

        # Re-cluster
        if self.clustering_method is not None:
            instances = self.clustering_method(int_points)
        else:
            clusterer = DBSCAN()
            instances = clusterer.fit_predict(int_points)

        unique_instances = np.unique(instances)

        # Re-Add objects
        for i in unique_instances:
            if i < 0:
                continue

            self._append_object_points(int_points[instances == i])

    def add_scan_points(self, scan, instance_labels=None):
        """
        Adds a whole scan of points as objects/features, clustered by their labels if provided.

        :param scan: (np.ndarray[nx3]) Array of xyz points to be added as individual features to the map.
        :param instance_labels: (np.ndarray[nx3]) Array of instance labels for each of the points. If None, then
                                the PointFeatureMap's clustering method will be used to group the scan points into
                                individual instances.

        :return: None
        """
        if instance_labels is None:
            if self.clustering_method is not None:
                instance_labels = self.clustering_method(scan)
            else:
                clusterer = DBSCAN()
                instance_labels = clusterer.fit_predict(scan)

        unique_instances = np.unique(instance_labels)

        for i in unique_instances:
            if i < 0:
                continue

            self.add_object_points(scan[instance_labels == i])
