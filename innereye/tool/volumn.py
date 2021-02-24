from collections import Iterable
import typing as t
from itertools import repeat
import importlib

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import napari

from ..lib.log import print_arguments
from ..lib.misc import local_arguments
from ..lib.img.misc import slide_over_ch, get_img_3d
from .base import MaskIO, Resetable
from .preprocessing import PreProcessing
from logging import getLogger
from ..lib.spots.call.blob import call_spots as call_blob

log = getLogger(__file__)


def func_for_slide(func: t.Callable, args: t.Tuple, channels: t.List) -> t.Callable:
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
        log.debug(f"Run function with args: {args_}")
        if ix_ch in channels:  # run only when channel specified.
            res = func(img, *args_)
        else:
            res = img
        if res.shape[1] <= 2:  # add z-axis to coordinates
            z = np.full((res.shape[0], 1), idx[1])
            res = np.c_[res, z]
        return res
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

    def view3d_mask(self, ixcy=[0, 1], ixch=[0, 1], label_mask=False):
        print_arguments(log.info)
        from skimage.measure import label
        if not isinstance(ixcy, list):
            ixcy = [ixcy]
        if not isinstance(ixch, list):
            ixch = [ixch]
        with napari.gui_qt():
            imgs4view = []
            channel_names = []
            for icy in ixcy:
                im4d = self.cycles[icy]
                for ich in ixch:
                    im3d_ch = im4d[:,:,:,ich]
                    im4view = self.__roll_im_for_view(im3d_ch)
                    imgs4view.append(im4view)
                    channel_names.append(f"signal cy:{icy} ch:{ich}")
            im4view = np.stack(imgs4view)
            viewer = napari.view_image(im4view, name=channel_names, channel_axis=0)
            for icy in ixcy:
                mask_cy = self.masks[icy]
                for ich in ixch:
                    mask_ch = mask_cy[:,:,:,ich]
                    mask4view = self.__roll_im_for_view(mask_ch)
                    if label_mask:
                        mask4view = label(mask4view)
                    label_layer = viewer.add_labels(mask4view, name=f"mask cy:{icy} ch:{ich}")
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


class Volumn(PreProcessing, ViewMask3D, MaskIO):
    """Deal with volumetric stuff.
    Puncta is something like spots but keep original pixel/voxel"""

    def __init__(self,
                 n_workers: int = 1,
                 record_num: int = 2):
        print_arguments(log.info)
        self.n_workers = n_workers
        self.cycles = None
        self.masks = None
        Resetable.__init__(self, ["cycles", "masks"], limit=record_num)

    def add_merged_cycle(self, merge_channel=False):
        print_arguments(log.info)
        for ixcy in range(1, len(self.cycles)):
            assert self.cycles[ixcy].shape == self.cycles[0].shape
        merged = sum(self.cycles) / len(self.cycles)
        if merge_channel:
            merged = merged.mean(axis=3, keepdims=True)
        self.set_new(self.cycles + [merged], "cycles")
        return self

    def call_mask(self, 
             p: float = 0.9,
             percentile_size: int = 15,
             q: float = 0.9,
             min_obj_size: int = 3,
             cycles=None,
             channels=None,
            ):
        print_arguments(log.info)
        masks = []
        if channels is None:
            channels = list(range(self.cycles[0].shape[-1]))
        if cycles is None:
            cycles = list(range(len(self.cycles)))
        def call_blob_(*args, return_blob=True):
            return call_blob(*args, return_blob=return_blob)
        call_blob_ = func_for_slide(call_blob_, (p, percentile_size, q, min_obj_size), channels)
        for ixcy, img in enumerate(self.cycles):
            if ixcy in cycles:
                blob = slide_over_ch(img, call_blob_, self.n_workers, stack=False)
                blob = np.stack(blob, axis=-1)
            else:
                blob = img
            masks.append(blob)
        self.set_new(masks, "masks")
        return self

    def mask_op(self,
                func_name="erosion",
                selm_shape="ball",
                selm_radius=1,
                cycles=None, channels=None
               ):
        mor = importlib.import_module("skimage.morphology")
        print_arguments(log.info)
        if channels is None:
            channels = list(range(self.cycles[0].shape[-1]))
        if cycles is None:
            cycles = list(range(len(self.cycles)))
        selm_func = getattr(mor, selm_shape)
        selm = selm_func(selm_radius)
        op_func = getattr(mor, func_name)
        process = func_for_slide(op_func, (selm,), channels)
        masks = []
        for ixcy, img in enumerate(self.masks):
            if ixcy in cycles:
                res = slide_over_ch(img, process, self.n_workers, stack=False)
                res = np.stack(res, axis=-1)
            else:
                res = img
            masks.append(res)
        self.set_new(masks, "masks")
        return self

