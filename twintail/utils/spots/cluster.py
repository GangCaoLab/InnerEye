import typing as t
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid
from skimage.morphology import dilation, ball, diamond
from skimage.measure import label, regionprops


def merge_close_points(points: np.ndarray,
                       min_dist: float = 1.5) -> np.ndarray:
    """Merge points close in space.

    :param points: Coordinates of all points before merge.
    :param min_dist: Min distance between points when perform clustering.
    :return: Coordinates of merged points.
    With the dtype np.float.
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
                          z_mode: str = 'slide') -> np.ndarray:
    """Merge points close in 3d euclidean space.

    :param points: Coordinates of all points before merge.
    In shape (n_points, 3).
    :param min_dist: Min distance between points when perform clustering.
    :param z_mode: How to deal with z-axis, {'slide', 'whole'}.
    :return: Coordinates of merged points.
    With the dtype np.float.
    """
    assert points.shape[1] == 3
    assert z_mode in {'slide', 'whole'}

    if z_mode == 'slide':
        z_layers = []
        for z in np.unique(points[:, 2]):
            pts = points[points[:, 2] == z]
            merged = merge_close_points(pts, min_dist)
            z_layers.append(merged)
        centroids = np.vstack(z_layers)
    else:
        centroids = merge_close_points(points, min_dist)

    return centroids


def coordinates_to_mask(points: np.ndarray,
                        shape: t.Optional[t.Tuple] = None) -> np.ndarray:
    """Convert coordinates to mask array.

    :param points: Coordinates of all points.
    Will be converted to the dtype np.int.
    :param shape: Shape of mask array.
    :return: Mask array with dtype bool.
    """
    points = points.astype(np.int)
    dim_max = tuple([points[:, i].max()+1 for i in range(points.shape[1])])
    if shape is None:
        shape = dim_max
    else:
        assert len(shape) == points.shape[1]
        shape = tuple([shape[i] or dim_max[i] for i in range(points.shape[1])])
    arr = np.zeros(shape, dtype=np.bool)
    ix = tuple(points[:, d] for d in range(points.shape[1]))
    arr[ix] = True
    return arr


def merge_close_points_cc(points: np.ndarray,
                          radius: int = 2) -> np.ndarray:
    """Merge close points by dilation and centroid of connected components.

    :param points: Coordinates of all points.
    In shape (n_points, 2) or (n_points, 3).
    Will be converted to the dtype np.int.
    :param radius: Min half distance between 2 point.
    :return: Coordinates of all points after merge.
    With the dtype np.float.
    """
    assert points.shape[1] in {2, 3}
    mask = coordinates_to_mask(points)
    if points.shape[1] == 2:
        se = diamond(radius)
    else:
        se = ball(radius)
    d = dilation(mask, se)
    ccs = regionprops(label(d))
    centroids = np.array([cc.centroid for cc in ccs])
    return centroids


def merge_close_points_3d_cc(points: np.ndarray,
                             radius: int = 2,
                             z_mode='slide') -> np.ndarray:
    """Merge close points by dilation and centroid of connected components.

    :param points: Coordinates of all points.
    In shape (n_points, 2) or (n_points, 3).
    Will be converted to the dtype np.int.
    :param radius: Min half distance between 2 point.
    :param z_mode: How to deal with z-axis, {'slide', 'whole'}.
    :return: Coordinates of all points after merge.
    With the dtype np.float.
    """
    assert points.shape[1] == 3
    assert z_mode in {'slide', 'whole'}

    if z_mode == 'slide':
        z_layers = []
        for z in np.unique(points[:, 2]):
            pts = points[points[:, 2] == z]
            merged = merge_close_points_cc(pts[:, :2], radius)
            merged = np.c_[merged, np.full(merged.shape[0], z)]
            z_layers.append(merged)
        centroids = np.vstack(z_layers)
    else:
        centroids = merge_close_points(points, radius)

    return centroids

