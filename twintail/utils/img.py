import typing as t

import numpy as np
from skimage.morphology import watershed
from skimage.measure import label
from pathos.multiprocessing import ProcessingPool as Pool


def cc_sub(im: np.ndarray, seed: np.ndarray, connectivity=2) -> np.ndarray:
    """Subtract the Connected Components in image which overlap with seed.

    :param im: mask image to be subtract CC.
    :param seed: mask image.
    :param connectivity: connectivity to calculate label, see:
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

    :return: CC in im without overlap with seed.
    """
    l = label(seed, connectivity=connectivity)
    w = watershed(im, markers=l, connectivity=connectivity, mask=im)
    o = w > 1
    d = im ^ o
    return d


def slide_over_z(
        arr: np.ndarray,
        func: t.Callable,
        n_workers: int = 1,
        stack_z: bool = True,
        stack_ch: bool = True) -> t.Union[np.ndarray, t.List]:
    """Map a function over z-axis.

    :param arr: Input image array, in shape (x, y, z, ch).
    :param func: Callable object used for map over array.
    Takes a 2d array as argument.
    :param n_workers: Number of cpus used.
    :param stack_z: Stack result along z-axis or not.
    :param stack_ch: Stack result along channel-axis or not.
    :return: Stacked array if stack == True,
    else a list of obj return by func split by channel.
    """
    idxs = [(ch, z) for ch in range(arr.shape[-1])
            for z in range(arr.shape[-2])]
    ims_2d = map(lambda t: arr[:, :, t[1], t[0]], idxs)
    pool = Pool(ncpus=n_workers)
    map_ = map if n_workers <= 1 else pool.imap
    chs = [[] for _ in range(arr.shape[-1])]
    for (ch, z), im in zip(idxs, map_(func, ims_2d, idxs)):
        chs[ch].append(im)
    if stack_z:
        chs = [np.stack(l, -1) for l in chs]
        if stack_ch:
            return np.stack(chs, -1)
        else:
            return chs
    else:
        return chs


def slide_over_ch(arr: np.ndarray,
                  func: t.Callable,
                  n_workers: int = 1,
                  stack: bool = True):
    """Map a function over channel-axis.

    :param arr: Input image array, in shape (x, y, z, ch).
    :param func: Callable object used for map over array.
    Takes a 3d array as argument.
    :param n_workers: Number of cpus used.
    :param stack: Stack result or not.
    :return: Stacked array if stack == True,
    else a list of obj return by func split by channel.
    """
    idxs = list(range(arr.shape[-1]))
    ims_3d = map(lambda i: arr[:, :, :, i], idxs)
    pool = Pool(ncpus=n_workers)
    map_ = map if n_workers <= 1 else pool.imap
    chs = []
    for ch, im in zip(idxs, map_(func, ims_3d, idxs)):
        chs.append(im)
    if stack:
        arr_ = np.stack(chs, -1)
        return arr_
    else:
        return chs

