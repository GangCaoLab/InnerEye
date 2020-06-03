import typing as t

import numpy as np
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from sklearn.neighbors import KDTree

from ..spots.call.lmpn import call_spots as call_peaks
from ..spots.cluster import merge_close_points_cc


def seg_watershed_2d(
        mask: np.ndarray,
        dist: np.ndarray,
        centers: np.ndarray,
        ) -> np.ndarray:
    seed = np.zeros(mask.shape)
    c_ = centers.astype(np.int)
    seed[c_[:, 0], c_[:, 1]] = np.arange(centers.shape[0]) + 1
    cells_masks = watershed(-dist, seed, mask=mask)
    return cells_masks


def otsu_watershed_2d(
        im2d: np.ndarray,
        gaussian_sigma: int = 8,
        min_cc_size: int = 500,
        merge_radius: int = 10,
        ) -> t.Tuple[np.ndarray, np.ndarray]:
    """Calculate cell center coordinates and mask,
    using otsu watershed method."""
    sg = ndi.gaussian_filter(im2d, gaussian_sigma)
    th = threshold_otsu(sg)
    mask = sg > th
    mask = remove_small_objects(mask, min_size=min_cc_size)
    dist = ndi.distance_transform_edt(mask)
    local_peaks = call_peaks(dist, neighbor_thresh=0.75)
    cells_center = merge_close_points_cc(local_peaks, merge_radius)
    cells_mask = seg_watershed_2d(mask, dist, cells_center)
    return cells_center, cells_mask


def cell_area_counts(labels: np.ndarray) -> t.Tuple[float, float]:
    regions = regionprops(labels)
    areas = np.array([r.area for r in regions])
    return areas.mean(), areas.std()


def gene2cell_2d(centers: np.ndarray,
                 mask: np.ndarray,
                 gene_pos: np.ndarray,
                 dist_th: float,
                 ) -> np.ndarray:
    """Assign gene to cells. 2d version
    :param centers: Cell's center coordinate. shape: [n, 2]
    :param mask: Cell's mask. shape: [w, h]
    :param gene_pos: All gene's coordinate. shape: [m, 2]
    :param dist_th: Distance(to cell's boundary) threshold.
    :return: All gene's corresponding cell's center. shape: [m, 2]
    """
    x_, y_ = np.where(mask > 0)
    pos_ = np.c_[x_, y_]
    tree = KDTree(pos_)
    dist, idx = tree.query(gene_pos)
    dist, idx = np.concatenate(dist), np.concatenate(idx)
    clost = pos_[idx, :]
    labels = mask[clost[:, 0], clost[:, 1]]
    res = centers[labels-1, :]
    res[dist > dist_th, :] = np.nan
    return res
