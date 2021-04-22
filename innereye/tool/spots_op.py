from itertools import repeat
from pathos.multiprocessing import ProcessingPool as Pool

from ..lib.log import print_arguments
from ..lib.spots.cluster import merge_close_points_3d, merge_close_points_3d_cc
from ..lib.spots.channel import (
    channel_merge, channel_merge_slidez, 
    filter_multi_channel, filter_multi_channel_slide_z,
)
from .base import ChainTool, SpotsIO


from logging import getLogger
log = getLogger(__file__)


class SpotsOp(ChainTool, SpotsIO):
    def __init__(self,
                 z_mode: str = 'slide',
                 n_workers: int = 1):
        print_arguments(log.info)
        assert z_mode in {'slide', 'whole'}
        self.z_mode = z_mode
        self.n_workers = n_workers
        self.dimensions = None
        self.spots = None
        self.combs = None

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

    def channel_merge(self,
                      radius: int = 2):
        """Perform channel merge."""
        print_arguments(log.info)
        spots = []
        combs = None
        if self.z_mode == 'slide':
            for spts in self.spots:
                spts_, combs = channel_merge_slidez(spts, radius, self.n_workers)
                spots.append(spts_)
        else:
            pool = Pool(ncpus=self.n_workers)
            map_ = map if self.n_workers == 1 else pool.map
            for spts_, combs in map_(channel_merge, self.spots, repeat(radius)):
                spots.append(spts_)
        self.combs = combs
        self.spots = spots
        return self

    def z_stack_spots(self):
        """Stack points along z-axis."""
        print_arguments(log.info)
        spots = []
        for spts in self.spots:
            new_spts = []
            for coords in spts:
                coords[:, 2] = 0
                new_spts.append(coords)
            spots.append(new_spts)
        self.spots = spots
        return self

    def filter_multi_channel(self, radius: float=2, n_thresh: int=2):
        """Filter out spots which has more than two channel neighbors"""
        print_arguments(log.info)
        spots = []
        if self.z_mode == 'slide':
            for spts in self.spots:
                spts_ = filter_multi_channel_slide_z(spts, radius, n_thresh, self.n_workers)
                spots.append(spts_)
        else:
            pool = Pool(ncpus=self.n_workers)
            map_ = map if self.n_workers == 1 else pool.map
            for spts_ in map_(filter_multi_channel, self.spots, repeat(radius), repeat(n_thresh)):
                spots.append(spts_)
        self.spots = spots
        return self

    read = SpotsIO.read_spots
    write = SpotsIO.write_spots
    count = SpotsIO.count_spots
