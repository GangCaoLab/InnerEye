import typing as t
import numpy as np
import scipy.sparse as ssp
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid


def merge_close_points(points: np.ndarray,
                       min_dist: float = 1.5) -> np.ndarray:
    """Merge points close in space.

    :param points: Coordinates of all points before merge.
    :param min_dist: Min distance between points when perform clustering.
    :return: Coordinates of merged points.
    """
    clustering = DBSCAN(min_dist, min_samples=1, metric='euclidean').fit(points)
    if np.unique(clustering.labels_).shape[0] > 1:
        centroids = NearestCentroid(metric='euclidean')\
            .fit(points, clustering.labels_)\
            .centroids_
    else:
        centroids = points
    return centroids


def merge_close_points_3d(points: np.ndarray,
                          min_dist: float = 1.5,
                          z_scale: float = 1.0,
                          z_mode: str = 'slide') -> np.ndarray:
    """Merge points close in 3d euclidean space.

    :param points: Coordinates of all points before merge.
    In shape (n_points, 3).
    :param min_dist: Min distance between points when perform clustering.
    :param z_mode: How to deal with z-axis, {'slide', 'whole'}.
    :param z_scale: Scale factor of the z-axis.
    :return: Coordinates of merged points.
    """
    assert points.shape[1] == 3
    assert z_mode in {'slide', 'whole'}
    if z_scale != 1.0:
        points = np.c_[[points[:, :2], points[:, 2]*z_scale]]

    if z_mode == 'slide':
        z_layers = []
        for z in np.unique(points[:, 2]):
            pts = points[points[:, 2] == z]
            merged = merge_close_points(pts, min_dist)
            z_layers.append(merged)
        centroids = np.vstack(z_layers)
    else:
        centroids = merge_close_points(points, min_dist)

    if z_scale != 1.0:
        non_zero = np.where(centroids[:, 2] != 0)
        centroids[non_zero] = centroids[non_zero] / z_scale

    return centroids


def coordinates_to_mask(points: np.ndarray,
                        shape: t.Optional[t.Tuple] = None) -> np.ndarray:
    """Convert coordinates to mask array.

    :param points: Coordinates of all points.
    In shape (n_points, 3) and dtype np.int.
    :param shape: Shape of mask array.
    :return: Mask array with dtype bool.
    """
    assert points.shape[1] == 3
    points = points.astype(np.int)
    dim_max = tuple([points[:, i].max()+1 for i in range(points.shape[1])])
    if shape is None:
        shape = dim_max
    else:
        assert len(shape) == points.shape[1]
        shape = tuple([shape[i] or dim_max[i] for i in range(points.shape[1])])
    arr = np.zeros(shape, dtype=np.bool)
    arr[points[:, 0], points[:, 1], points[:, 2]] = True
    return arr
