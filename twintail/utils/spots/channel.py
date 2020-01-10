from itertools import combinations, repeat
import typing as t
from functools import reduce
from operator import eq

import numpy as np
from skimage.morphology import watershed
from skimage.morphology import diamond, ball, dilation
from skimage.measure import label
from pathos.multiprocessing import ProcessingPool as Pool

from .misc import coordinates_to_mask, cc_centroids


def channel_combinations(ch_idxs: t.List[int],
                         num: int,
                         dangling: bool = True) -> t.List[t.List[int]]:
    """Make channel combination.

    :param ch_idxs: Indexes of channels.
    :param num: Number of channel per combination.
    :param dangling: If True consider the dangling case,
    like (0, 0), (1, 1) ...
    :return:
    """
    combs = [list(c) for c in list(combinations(ch_idxs, num))]
    if dangling:
        combs += [[i for _ in range(num)] for i in ch_idxs]
    return combs


def cc_sub(im: np.ndarray, seed: np.ndarray, connectivity=2) -> np.ndarray:
    """Subtract the Connected Components in image which overlap with seed.

    :param im: mask image to be subtract CC.
    :param seed: mask image.
    :param connectivity: connectivity to calculate label, see:
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

    :return: CC in im without overlap with seed.
    """
    lb = label(seed, connectivity=connectivity)
    w = watershed(im, markers=lb, connectivity=connectivity, mask=im)
    o = w > 1
    d = im ^ o
    return d


def mask_merge(mask1: np.ndarray,
               mask2: np.ndarray,
               radius: int,
               ) -> np.ndarray:
    assert mask1.shape == mask2.shape
    dim = len(mask1.shape)
    se = diamond(radius) if dim == 2 else ball(radius)
    mask1 = dilation(mask1, se)
    mask2 = dilation(mask2, se)
    common = mask1 & mask2
    return common


def mask_sub(oriangal: np.ndarray,
             masks: t.List[np.ndarray],
             ) -> np.ndarray:
    o = oriangal
    for m in masks:
        o = cc_sub(o, m)
    return o


def channel_merge(spots: t.List[np.ndarray],
                  radius: int = 2,
                  ) -> t.Tuple[t.List[np.ndarray],
                               t.List[t.List[int, int]]]:
    # input checks
    assert reduce(eq, map(lambda pts: pts.shape[1], spots))
    dim = spots[0].shape[1]
    assert 2 <= dim <= 3
    # convert coordinates to int type for convert to mask(dense matrix)
    spots = [pts.astype(np.int) for pts in spots]
    shape = tuple([max([pts[:, i] for pts in spots])
                   for i in range(dim)])
    masks = [coordinates_to_mask(pts, shape) for pts in spots]
    combs = channel_combinations(list(range(len(spots))), 2, dangling=True)

    res_masks = []
    def contain(idx):
        # select all combination's merged mask related to given index
        return [res_masks[i] for i, c in enumerate(combs)
                if (idx in c) and (c[0] != c[1])]

    for i, j in combs:
        if i != j:
            merged = mask_merge(masks[i], masks[j], radius)
            res_masks.append(merged)
        else:  # dangling
            sub = mask_sub(masks[i], contain(i))
            res_masks.append(sub)

    results = [cc_centroids(m) for m in res_masks]

    return results, combs


def channel_merge_slidez(spots: t.List[np.ndarray],
                         radius: int = 2,
                         n_workers: int = 1,
                         ) -> t.Tuple[t.List[np.ndarray],
                                      t.List[t.List[int, int]]]:
    assert all([pts.shape[1] == 3 for pts in spots])
    zs = np.unique(np.r_[[pts[:, 2] for pts in spots]])
    spots_by_z = []
    for z in zs:  # split by z
        spots_z = []
        for pts in spots:
            in_layer = pts[pts[:, 2] == z]
            spots_z.append(in_layer)
        spots_by_z.append(spots_z)

    # slide processing each z layer
    pool = Pool(ncpus=n_workers)
    map_ = map if n_workers <= 1 else pool.map
    res_by_z = []
    combs = None
    for res, combs in map_(channel_merge, spots_by_z, repeat(radius)):
        res_by_z.append(res)

    merged = []  # merge by z
    for z, res in zip(zs, res_by_z):
        for idx, pts in enumerate(res):
            pts_ = np.c_[pts, np.full(pts.shape[0], z)]
            try:
                merged[idx] = np.r_[merged[idx], pts_]
            except IndexError:
                merged.append(pts_)
    return merged, combs

