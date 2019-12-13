import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid


def merge_close_points(points: np.ndarray,
                       eps: float = 1) -> np.ndarray:
    """Merge points close in space.

    :param points: Coordinates of all points before merge.
    :param eps: Min distance between points when perform clustering.
    :return: Coordinates of merged points.
    """
    clustering = DBSCAN(eps, min_samples=1, metric='euclidean').fit(points)
    centroids = NearestCentroid(metric='euclidean')\
        .fit(points, clustering.labels_)\
        .centroids_
    return centroids

