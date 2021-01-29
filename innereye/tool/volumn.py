from collections import Iterable
import typing as t
from itertools import repeat

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import napari

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
        log.debug(f"Run blob call function with args: {args_}")
        blobs = func(img, *args_)
        if blobs.shape[1] <= 2:  # add z-axis to coordinates
            z = np.full((blobs.shape[0], 1), idx[1])
            blobs = np.c_[blobs, z]
        return blobs
    return wrap


class ViewMask3D(object):
    @staticmethod
    def __roll_im_for_view(im):
        m = np.rollaxis(im, 2, 0)
        return m

    def view3d(self, ixcy=[0, 1], ixch=[0, 1]):
        print_arguments(log.info)
        if not isinstance(ixcy, list):
            ixcy = [ixcy]
        if not isinstance(ixch, list):
            ixch = [ixch]
        for_view = []
        channel_names = []
        for icy in ixcy:
            im4d = self.cycles[icy]
            mask_cy = self.masks[icy]
            for ich in ixch:
                im3d_ch = im4d[:,:,:,ich]
                mask_ch = mask_cy[ich]
                im4view = self.__roll_im_for_view(im3d_ch)
                for_view.append(im4view)
                channel_names.append(f"signal cy:{icy} ch:{ich}")
                mask4view = self.__roll_im_for_view(mask_ch)
                for_view.append(mask4view)
                channel_names.append(f"mask cy:{icy} ch:{ich}")
        for_view = np.stack(for_view)
        napari.view_image(for_view, channel_axis=0, name=channel_names)
        return self


class Puncta(PreProcessing, ViewMask3D):
    """Puncta is something like spots but keep original pixel/voxel"""

    def __init__(self,
                 n_workers: int = 1):
        print_arguments(log.info)
        self.n_workers = n_workers
        self.cycles = None
        self.masks = None
        Resetable.__init__(self, "cycles")

    def blob(self, 
             p: float = 0.9,
             percentile_size: int = 15,
             q: float = 0.9,
             min_obj_size: int = 3,
            ):
        print_arguments(log.info)
        masks = []
        def call_blob_(*args, return_blob=True):
            return call_blob(*args, return_blob=return_blob)
        call_blob_ = func_for_slide(call_blob_, (p, percentile_size, q, min_obj_size))
        for img in self.cycles:
            blob = slide_over_ch(img, call_blob_, self.n_workers, stack=False)
            masks.append(blob)
        self.masks = masks
        return self

