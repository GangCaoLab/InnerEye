from itertools import combinations, repeat
import typing as t
from functools import reduce
from operator import eq

import numpy as np
from skimage.morphology import watershed
from skimage.morphology import diamond, ball, dilation
from skimage.measure import label
from sklearn.neighbors import KDTree
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
                               t.List[t.List[int]]]:
    # input checks
    assert all([pts.shape[1] == spots[0].shape[1] for pts in spots[1:]])
    dim = spots[0].shape[1]
    assert 2 <= dim <= 3
    # convert coordinates to int type for convert to mask(dense matrix)
    spots = [pts.astype(np.int) for pts in spots]
    shape = tuple([max([pts[:, i].max() for pts in spots]) + 1
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


def split_by_z(spots: t.List[np.ndarray]) -> t.Tuple[t.List[np.ndarray], np.ndarray]:
    assert all([pts.shape[1] == 3 for pts in spots])
    zs = np.unique(np.hstack([pts[:, 2] for pts in spots]))
    spots_by_z = []
    for z in zs:  # split by z
        spots_z = []
        for pts in spots:
            in_layer = pts[pts[:, 2] == z][:, :2]
            spots_z.append(in_layer)
        spots_by_z.append(spots_z)
    return spots_by_z, zs
    

def merge_by_z(res_by_z, zs) -> t.List[np.ndarray]:
    merged = []  # merge by z
    for z, res in zip(zs, res_by_z):
        for idx, pts in enumerate(res):
            pts_ = np.c_[pts, np.full(pts.shape[0], z)]
            try:
                merged[idx] = np.r_[merged[idx], pts_]
            except IndexError:
                merged.append(pts_)
    return merged


def channel_merge_slidez(spots: t.List[np.ndarray],
                         radius: int = 2,
                         n_workers: int = 1,
                         ) -> t.Tuple[t.List[np.ndarray],
                                      t.List[t.List[int]]]:
    spots_by_z, zs = split_by_z(spots)

    # slide processing each z layer
    pool = Pool(ncpus=n_workers)
    map_ = map if n_workers <= 1 else pool.map
    res_by_z = []
    combs = None
    for res, combs in map_(channel_merge, spots_by_z, repeat(radius)):
        res_by_z.append(res)

    merged = merge_by_z(res_by_z, zs)
    return merged, combs


def channel_merge_kdtree(
    spots: t.List[np.ndarray],
    radius: float = 2,
    ) -> t.Tuple[t.List[np.ndarray],
                 t.List[t.List[int]]]:
    # input checks
    assert all([pts.shape[1] == spots[0].shape[1] for pts in spots[1:]])
    dim = spots[0].shape[1]
    assert 2 <= dim <= 3

    d_lim = radius * 2

    combs = channel_combinations(list(range(len(spots))), 2, dangling=True)
    results = []
    trees = [KDTree(ch) for ch in spots]
    used_idx = [[] for _ in spots]
    for i, j in combs:
        if i == j: continue
        tree = trees[i]
        ch1, ch2 = spots[i], spots[j]
        d, ix1_ = tree.query(ch2)
        lt_d = d <= d_lim
        ix1 = ix1_[lt_d]
        pts1 = ch1[ix1]
        ix2 = np.where(lt_d)[0]
        pts2 = ch2[ix2]
        pts = (pts1 + pts2) / 2
        results.append(pts)
        used_idx[i].append(ix1)
        used_idx[j].append(ix2)
    for i, idxs in enumerate(used_idx):
        idx = np.unique(np.concatenate(idxs))
        used_idx[i] = idx
    for i, j in combs:
        if i != j: continue
        ch = spots[i]
        idx_used = used_idx[i]
        mask = np.full(ch.shape[0], True)
        mask[idx_used] = False
        idx_unused = np.where(mask)[0]
        pts = ch[idx_unused]
        results.append(pts)
    return results, combs


def filter_multi_channel(spots: t.List[np.ndarray], radius: float, n_thresh: int=2) -> t.List[np.ndarray]:
    spots_ = []
    for query_ix, pts in enumerate(spots):  # pts shape: [n, 3] or [n, 2]
        other_channel_index = []
        other_channel = []
        for ix in range(len(spots)):
            if ix == query_ix:
                continue
            other_channel.append(spots[ix])
            other_channel_index.append(np.ones(spots[ix].shape[0])*ix)
        ref_pts = np.concatenate(other_channel)
        ref_index = np.concatenate(other_channel_index)
        tree = KDTree(ref_pts)
        ind_per_query = tree.query_radius(pts, radius)
        mask_multi_c = np.array([
            np.unique(ref_index[ind]).shape[0] >= n_thresh
            for ind in ind_per_query
        ])
        new_pts = pts[~mask_multi_c]
        spots_.append(new_pts)
    return spots_


def filter_multi_channel_slide_z(
    spots: t.List[np.ndarray],
    radius: float,
    n_thresh: int=2,
    n_workers: int = 1,
    ) -> t.List[np.ndarray]:
    spots_by_z, zs = split_by_z(spots)

    # slide processing each z layer
    pool = Pool(ncpus=n_workers)
    map_ = map if n_workers <= 1 else pool.map
    res_by_z = []
    for res in map_(filter_multi_channel, spots_by_z, repeat(radius), repeat(n_thresh)):
        res_by_z.append(res)

    merged = merge_by_z(res_by_z, zs)
    return merged

