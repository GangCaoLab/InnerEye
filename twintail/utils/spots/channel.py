from itertools import combinations
import typing as t

import numpy as np
from skimage.morphology import watershed
from skimage.measure import label

from .misc import coordinates_to_mask


def channel_combinations(ch_idxs: t.List[int],
                         num: int,
                         dangling: bool = True) -> t.List[t.Tuple]:
    """Make channel combination.

    :param ch_idxs: Indexes of channels.
    :param num: Number of channel per combination.
    :param dangling: If True consider the dangling case,
    like (0, 0), (1, 1) ...
    :return:
    """
    combs = list(combinations(ch_idxs, num))
    if dangling:
        combs += [tuple([i for _ in range(num)]) for i in ch_idxs]
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


def spots_merge(points1: np.ndarray,
                points2: np.ndarray,
                radius: int)\
        -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert points1.shape[1] == points2.shape[1]
    dim = points1.shape[1]
    shape = tuple([max(points1[:, i], points2[:, i])+1 for i in range(dim)])
    mask1 = coordinates_to_mask(points1, shape)
    mask2 = coordinates_to_mask(points2, shape)
    pass


def channel_merge():
    pass

