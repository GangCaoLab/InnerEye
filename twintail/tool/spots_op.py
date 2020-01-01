from pathos.multiprocessing import ProcessingPool as Pool
from twintail.utils.log import print_arguments
from twintail.utils.spots.cluster import merge_close_points_3d, merge_close_points_3d_cc
from .base import SpotsTool


from logging import getLogger
log = getLogger(__file__)


class SpotsOp(SpotsTool):
    def __init__(self,
                 z_mode: str = 'slide',
                 n_workers: int = 1):
        print_arguments(log.info)
        assert z_mode in {'slide', 'whole'}
        self.z_mode = z_mode
        self.n_workers = n_workers
        self.dimensions = None
        self.spots = None

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
