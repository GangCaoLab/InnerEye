import typing as t
import numpy as np
from ..lib.log import print_arguments
from ..lib.spots.call import lmpn, blob, tophat_extrema
from ..lib.img.misc import slide_over_z, slide_over_ch
from ..lib.misc import local_arguments
from .base import ChainTool, ImgIO, SpotsIO


from logging import getLogger
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


def call_spots(func: t.Callable,
               cycles: t.List[np.ndarray],
               z_mode: str,
               n_workers: int) -> t.List[t.List[np.ndarray]]:
    if z_mode == 'slide':
        spots = [slide_over_z(img, func, n_workers, stack_z=False, stack_ch=False)
                 for img in cycles]
        spots = [[np.vstack(ch) for ch in cy] for cy in spots]
    else:
        spots = [slide_over_ch(img, func, n_workers, stack=False)
                 for img in cycles]
    return spots


class CallSpots(ChainTool, ImgIO, SpotsIO):
    """Find spots(signal) within image. """
    def __init__(self,
                 z_mode: str = 'slide',
                 n_workers: int = 1):
        print_arguments(log.info)
        assert z_mode in {'slide', 'whole'}
        self.z_mode = z_mode
        self.n_workers = n_workers
        self.cycles = None
        self.dimensions = None
        self.spots = None

    def lmpn(self,
             maximum_filter_size=5,
             percentile_filter_size=11,
             percentile=80.0,
             neighbor_size=3,
             neighbor_thresh=0.5):
        """Call spots using LMPN method."""
        print_arguments(log.info)
        args = local_arguments(keywords=False)
        func = func_for_slide(lmpn.call_spots, args)
        self.spots = call_spots(func, self.cycles, self.z_mode, self.n_workers)
        return self

    def blob(self,
             p=0.9,
             percentile_size=15,
             q=0.9,
             min_obj_size=3):
        """Call spots using blob method."""
        print_arguments(log.info)
        args = local_arguments(keywords=False)
        func = func_for_slide(blob.call_spots, args)
        self.spots = call_spots(func, self.cycles, self.z_mode, self.n_workers)
        return self

    def tophat_extrema(self,
                       h=0.1,
                       q=None):
        """Call spots using tophat-filter + h_maxima method."""
        print_arguments(log.info)
        args = local_arguments(keywords=False)
        func = func_for_slide(tophat_extrema.call_spots, args)
        self.spots = call_spots(func, self.cycles, self.z_mode, self.n_workers)
        return self

    read = ImgIO.read_img
    write = SpotsIO.write_spots
    count = SpotsIO.count_spots
