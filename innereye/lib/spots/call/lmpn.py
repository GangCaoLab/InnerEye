from numba import jit
import scipy.ndimage as ndi
import numpy as np


@jit(nopython=True)
def count_neighbors_2d(points, array, size=3):
    s_ = size // 2
    counts = np.zeros(points.shape[0], dtype=np.uint8)
    for idx in range(points.shape[0]):
        center_pos = points[idx]
        c = 0
        x, y = center_pos
        local = array[max(x-s_, 0):x+s_+1, max(y-s_, 0):y+s_+1]
        for i in range(local.shape[0]):
            for j in range(local.shape[1]):
                if local[i, j] and (i, j) != (x, y):
                    c += 1
        counts[idx] = c
    return counts


@jit(nopython=True)
def count_neighbors_3d(points, array, size=3):
    s_ = size // 2
    counts = np.zeros(points.shape[0], dtype=np.uint8)
    for idx in range(points.shape[0]):
        center_pos = points[idx]
        c = 0
        x, y, z = center_pos
        local = array[max(x-s_, 0):x+s_+1, max(y-s_, 0):y+s_+1, max(z-s_, 0):z+s_+1]
        for i in range(local.shape[0]):
            for j in range(local.shape[1]):
                for k in range(local.shape[2]):
                    if local[i, j, k] and (i, j, k) != (x, y, z):
                        c += 1
        counts[idx] = c
    return counts


def call_spots(roi: np.ndarray,
               maximum_filter_size: int = 5,
               percentile_filter_size: int = 11,
               percentile: float = 80.0,
               neighbor_size: int = 3,
               neighbor_thresh: float = 0.5,
               ) -> np.ndarray:
    roi = roi.astype(np.float32)
    maxi = ndi.maximum_filter(roi, size=maximum_filter_size)
    local_maxi = (maxi - roi) == 0
    per = ndi.percentile_filter(roi, percentile, size=percentile_filter_size)
    local_per = (roi - per) >= 0
    candidates = np.c_[np.where(local_maxi & local_per)]
    if candidates.shape[1] == 2:
        ns = count_neighbors_2d(candidates, local_per, neighbor_size)
    else:
        ns = count_neighbors_3d(candidates, local_per, neighbor_size)
    n_thresh = int(neighbor_thresh * (neighbor_size ** 2))
    pts = candidates[ns > n_thresh]
    return pts
