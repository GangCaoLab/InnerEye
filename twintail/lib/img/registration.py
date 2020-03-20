import os
import typing as t
from logging import getLogger

import numpy as np
import SimpleITK as sitk

from .misc import slide_over_z

log = getLogger(__file__)


def get_elastix_log_dir():
    basedir = "./"
    names = os.listdir(basedir)
    ix = 0
    name = lambda: f"elastix.log.{ix}"
    while name() in names: ix += 1
    os.mkdir(name())
    return name()


def get_img_2d(im4d: np.ndarray,
               ch: t.Union[str, int],
               z: t.Union[str, int]) -> np.ndarray:
    assert (ch == 'mean') or (type(ch) is int)
    assert (z == 'mean') or (type(z) is int)
    if ch == 'mean':
        im3d = im4d.mean(axis=3)
    else:
        im3d = im4d[:, :, :, ch]
    if z == 'mean':
        im2d = im3d.mean(axis=2)
    else:
        im2d = im3d[:, :, z]
    return im2d


class Registration2d(object):

    def __init__(self,
                 cycles: t.List[np.ndarray],
                 ref_cycle: int = -1,
                 ref_channel: t.Union[int, str] = 'mean',
                 ref_z: t.Union[int, str] = 'mean',
                 n_workers: int = 1,
                 ):
        self.cycles = cycles
        self.ref_cycle = ref_cycle
        self.ref_channel = ref_channel
        self.ref_z = ref_z
        self.selx = sitk.ElastixImageFilter()
        self.selx.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        self.transforms = None
        self.n_workers = n_workers

        self.selx.LogToFileOn()
        self.selx.SetOutputDirectory(get_elastix_log_dir())
        self.selx.LogToConsoleOff()

    def estimate_transform(self):
        ix_ref = self.ref_cycle
        selx = self.selx
        ref_im4d = self.cycles[ix_ref]
        ref_im2d = get_img_2d(ref_im4d, self.ref_channel, self.ref_z)
        ref_itk = sitk.GetImageFromArray(ref_im2d)
        selx.SetFixedImage(ref_itk)
        transforms = []
        for idx, im4d in enumerate(self.cycles):
            if (idx == ix_ref) or (idx == (len(self.cycles) + ix_ref)):
                trans = None
            else:
                im2d = get_img_2d(im4d, self.ref_channel, self.ref_z)
                im_itk = sitk.GetImageFromArray(im2d)
                selx.SetMovingImage(im_itk)
                selx.Execute()
                trans = selx.GetTransformParameterMap()
            transforms.append(trans)
        self.transforms = transforms

    def apply(self) -> t.List[np.ndarray]:
        ix_ref = self.ref_cycle
        aligned = []
        for idx, im4d in enumerate(self.cycles):
            if (idx == ix_ref) or (idx == (len(self.cycles) + ix_ref)):
                res = im4d
            else:
                trans = self.transforms[idx]
                def proc_im2d(im2d, *args):
                    im_itk = sitk.GetImageFromArray(im2d)
                    im_res = sitk.Transformix(im_itk, trans)
                    return sitk.GetArrayFromImage(im_res)
                res = slide_over_z(im4d, proc_im2d, self.n_workers)
            aligned.append(res)
        return aligned
