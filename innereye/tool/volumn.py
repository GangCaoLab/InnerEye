from collections import Iterable
import typing as t
from itertools import repeat

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import napari

from ..lib.log import print_arguments
from ..lib.misc import local_arguments
from ..lib.img.misc import slide_over_ch, get_img_3d
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
    def __roll_im_for_view(im, dim=3):
        if dim == 3:
            m = np.rollaxis(im, 2, 0)
        elif dim == 4:
            m = np.rollaxis(im, 3, 0)
            m = np.rollaxis(m, 3, 1)
        else:
            raise ValueError(f"Only support 3 and 4 dimension.")
        return m

    def view3d_signal(self, ixcy=[0], ixch=[0,1,2,3], merge_ch=False):
        print_arguments(log.info)
        if not isinstance(ixcy, list):
            ixcy = [ixcy]
        for_view = []
        channel_names = []
        for icy in ixcy:
            cy = self.cycles[icy]
            if merge_ch:
                imch = get_img_3d(cy, ixch)[:, :, :, np.newaxis]
                cy4view = self.__roll_im_for_view(imch, dim=4)
                for_view.append(cy4view)
                channel_names.append(f"cy:{icy} ch:{ixch}")
            else:
                imch = cy[:,:,:,ixch]
                cy4view = self.__roll_im_for_view(imch, dim=4)
                for_view.append(cy4view)
                channel_names.extend([f"cy:{icy} ch:{ich}" for ich in ixch])
        for_view = np.concatenate(for_view)
        napari.view_image(for_view, channel_axis=0, name=channel_names)
        return self

    def view3d_mask(self, ixcy=[0, 1], ixch=[0, 1]):
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

    def view_mean_along_z(self):
        print_arguments(log.info)
        im = []
        for icy, im4d in enumerate(self.cycles):
            imcy = im4d.mean(axis=0).mean(axis=0).mean(axis=1)
            im.append(imcy)
        im = np.stack(im)
        napari.view_image(im, name="mean along z")
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


