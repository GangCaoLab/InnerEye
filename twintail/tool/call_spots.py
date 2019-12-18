import typing as t
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from twintail.utils.log import print_arguments
from twintail.utils.spots.call import lmpn, blob, tophat_extrema
from twintail.utils.spots.cluster import merge_close_points_3d, merge_close_points_3d_cc
from twintail.utils.img import slide_over_z, slide_over_ch
from twintail.utils.misc import local_arguments
from twintail.utils.io.h5 import write_spots
from .base import ChainTool


from logging import getLogger
log = getLogger(__file__)


def func_for_slide(func: t.Callable, args: t.Tuple) -> t.Callable:
    """Construct the function for slide over whole image."""
    def wrap(img: np.ndarray,
             idx: t.Union[int, t.Tuple[int, int]]) -> np.ndarray:
        spots = func(img, *args)
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


class CallSpots(ChainTool):
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

    def write(self, path: str):
        """Write spot coordinates to disk"""
        print_arguments(log.info)
        dims = [dim[:3] for dim in self.dimensions]
        write_spots(path, self.spots, dims)
        return self

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

    def merge_neighboring(self,
                          min_dist=2.0,
                          method="dilation"):
        """Merge neighboring points."""
        print_arguments(log.info)
        assert method in {'dbscan', 'dilation'}
        if method == 'dbscan':
            merge_func = merge_close_points_3d
        else:
            min_dist = min_dist // 2
            merge_func = merge_close_points_3d_cc

        pool = Pool(ncpus=self.n_workers)
        map_ = map if self.n_workers <= 1 else pool.imap
        idx = [(ixcy, ixch)
               for ixcy in range(len(self.spots))
               for ixch in range(len(self.spots[ixcy]))]
        coords = map_(lambda t: self.spots[t[0]][t[1]], idx)
        func = lambda c: merge_func(c, min_dist, self.z_mode)
        spots = [[] for _ in range(len(self.spots))]
        for (ixcy, ixch), im in zip(idx, map_(func, coords)):
            spots[ixcy].append(im)
        self.spots = spots
        return self

    def count(self, outfile=None, z=True):
        """Count number of points in each cycle and channel.

        :param outfile: Write count result to specified file.
        :param z: Count each z layer or not.
        :return:
        """
        print_arguments(log.info)
        msg = ""
        for ixcy, chs in enumerate(self.spots):
            msg += f"Cycle index: {ixcy}\n"
            for ixch, coords in enumerate(chs):
                msg += f"\tChannel index: {ixch}\n"
                if z:
                    for z in np.unique(coords[:, 2]):
                        layer = coords[coords[:, 2] == z]
                        msg += f"\t\t{z}\t{layer.shape[0]}\n"
                else:
                    msg += f"\t\t{coords.shape[0]}\n"
        log.info(msg)
        if outfile:
            with open(outfile, 'w') as f:
                f.write(msg)
        return self

