import typing as t

import numpy as np
from skimage.morphology import watershed
from skimage.measure import label


def cc_sub(im: np.ndarray, seed: np.ndarray, connectivity=2) -> np.ndarray:
    """Substract the Connected Components in image which overlaped with seed.

    :param im1: mask image to be subtract CC.
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


def slide_over_ch_z(
        arr: np.ndarray,
        func: t.Callable,
        n_workers: int = 1,
        stack=True) -> t.Union[np.ndarray, t.List[t.List[t.Any]]]:
    from pathos.multiprocessing import ProcessingPool as Pool
    idxs = [(ch, z) for ch in range(arr.shape[-1])
            for z in range(arr.shape[-2])]
    ims_2d = map(lambda t: arr[:, :, t[1], t[0]], idxs)
    pool = Pool(ncpus=n_workers)
    map_ = map if n_workers <= 1 else pool.imap
    chs = [[] for _ in range(arr.shape[-1])]
    for (ch, z), im in zip(idxs, map_(func, ims_2d, idxs)):
        chs[ch].append(im)
    if stack:
        chs = [np.stack(l, -1) for l in chs]
        arr_ = np.stack(chs, -1)
        return arr_
    else:
        return chs
