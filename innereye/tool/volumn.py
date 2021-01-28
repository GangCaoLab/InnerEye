from collections import Iterable
import typing as t
from itertools import repeat

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from ..lib.log import print_arguments
from ..lib.misc import local_arguments
from ..lib.img.misc import slide_over_ch
from .base import ChainTool, ImgIO, Resetable
from .preprocessing import PreProcessing
from logging import getLogger
from ..lib.spots.call.blob import call_spots as call_blob

log = getLogger(__file__)


def func_for_slide(func: t.Callable, args: t.Tuple) -> t.Callable:
    """Construct the function for slide over whole image."""
    def wrap(img: np.ndarray,
             idx: t.Union[int, t.Tuple[int, int]]) -> np.ndarray:

        # split args to different channels
        ix_ch = idx if not isinstance(idx, tuple) else idx[0]
        args_ = []
        for a in args:
            if isinstance(a, list):
                p = a[ix_ch]
            else:
                p = a
            args_.append(p)
        log.debug(f"Run spots calling function with args: {args_}")
        spots = func(img, *args_)
        if spots.shape[1] <= 2:  # add z-axis to coordinates
            z = np.full((spots.shape[0], 1), idx[1])
            spots = np.c_[spots, z]
        return spots
    return wrap


class Puncta(PreProcessing):
    """Puncta is something like spots but keep original pixel/voxel"""

    def __init__(self,
                 n_workers: int = 1):
        print_arguments(log.info)
        self.n_workers = n_workers
        self.cycles = None
        self.masks = None

    def blob(self, 
             p: float = 0.9,
             percentile_size: int = 15,
             q: float = 0.9,
             min_obj_size: int = 3,
            ):
        masks = []
        def call_blob_(*args):
            return call_blob(*args[:-1], True)
        call_blob_ = func_for_slide(call_blob_, (p, percentile_size, q, min_obj_size))
        for img in self.cycles:
            blob = slide_over_ch(img, call_blob_, self.n_workers, stack=False)
            masks.append(blob)
        self.masks = masks

