from itertools import combinations
import typing as t

import numpy as np
import scipy.ndimage as ndi
from pathos.multiprocessing import ProcessingPool as Pool

from twintail.utils.io.h5 import read_cycles, write_cycles
from twintail.utils.log import print_arguments
from twintail.utils.img import cc_sub, slide_over_ch_z
from twintail.utils import flatten

from logging import getLogger
log = getLogger(__file__)


class SignalCall(object):
    """Signal points call."""
    def __init__(self,
                 z: str = 'slide',
                 signal_channels: t.List[int] = (0, 1, 2, 3),
                 channels_per_cycle: int = 2,
                 n_workers: int = 1):
        print_arguments(log.info)
        assert z in {'slide'}
        self.signal_channels = list(signal_channels)
        self.channels_per_cycle = channels_per_cycle
        self.n_workers = n_workers

    def read(self, path: str):
        print_arguments(log.info)
        self.cycles = read_cycles(path)
        self.cycles = [arr[:, :, :, self.signal_channels] for arr in self.cycles]
        return self

    def candidate(self, percentile=80, size=11, q=0.90, min_size=3, min_neighbors=3):
        print_arguments(log.info)
        from skimage.morphology import remove_small_objects

        selm = np.ones((3,3))

        def process(im, *args):
            p = ndi.percentile_filter(im, percentile, size)
            t = np.quantile(im, q)
            sig = (im > p) & (im > t)
            f_sig = remove_small_objects(sig, min_size=min_size, connectivity=1)
            f_sig = f_sig & (ndi.convolve(f_sig.astype(np.int), selm) > min_neighbors)
            f_sig = remove_small_objects(f_sig, min_size=min_size, connectivity=1)
            return f_sig

        self.signals = []
        for arr_ in map(lambda arr: slide_over_ch_z(arr, process, self.n_workers), self.cycles):
            self.signals.append(arr_)

        return self

    @property
    def channel_combinations(self):
        combs = list(combinations(self.signal_channels, self.channels_per_cycle))
        return combs

    def channel_merge(self):
        print_arguments(log.info)
        merged = []
        for ix_cy, arr in enumerate(self.signals):
            chs = []
            for ix1, ix2 in self.channel_combinations:
                a = arr[:, :, :, ix1]
                b = arr[:, :, :, ix2]
                ab = a & b
                chs.append(ab)
            merged.append(np.stack(chs, axis=-1))
        self.merged = merged
        return self

    def call_dangling(self):
        """Call the signals of dangling(AA TT CC GG..)"""
        print_arguments(log.info)
        combs = self.channel_combinations

        def process_ch(args):
            arr, a, bs = args
            for bix in bs:
                b = arr[:, :, :, bix]
                zs = []
                for ix_z in range(b.shape[-1]):
                    a_ = a[:,:,ix_z]
                    b_ = b[:,:,ix_z]
                    a_sub_b = cc_sub(a_, b_)
                    zs.append(a_sub_b)
                a = np.stack(zs, axis=-1)
            return a

        def gen_args(arr):
            for ix in self.signal_channels:
                a = self.signals[ix_cy][:, :, :, ix]
                bs = [i for i, t in enumerate(combs) if ix in t]
                yield arr, a, bs

        pool = Pool(ncpus=self.n_workers)
        map_ = pool.imap if self.n_workers > 1 else map

        for ix_cy, arr in enumerate(self.merged):
            chs = []
            for arr_ in map_(process_ch, gen_args(arr)):
                chs.append(arr_)
            danglings = np.stack(chs, axis=-1)
            self.merged[ix_cy] = np.c_[arr, danglings]
        return self

    def write_merged(self, path):
        print_arguments(log.info)
        write_cycles(path, self.merged)
        return self

    def call_merged_centroid(self):
        print_arguments(log.info)
        from skimage.measure import label, regionprops

        def process(im, ix):
            ix_z, ix_ch = ix
            l = label(im, connectivity=2)
            centroids = []
            for r in regionprops(l):
                y, x = r.centroid
                y, x = int(y), int(x)
                centroids.append([y, x, ix_z, ix_ch])
            centroids = np.array(centroids)
            return centroids

        self.points = []
        for ix_cy, arr in enumerate(self.merged):
            chs = slide_over_ch_z(
                arr, process, n_workers=self.n_workers, stack=False)
            centroids = np.vstack(list(filter(lambda a: a.shape[0] > 0, flatten(chs))))
            self.points.append(centroids)
        return self

    def write_points(self, path):
        print_arguments(log.info)
        write_cycles(path, self.points)
        return self


if __name__ == "__main__":
    from twintail.utils.log import set_global_logging
    set_global_logging()
    import fire
    fire.Fire(SignalCall)
