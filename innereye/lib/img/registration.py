import os
import typing as t
from logging import getLogger
import abc

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi

from .misc import get_img_2d, get_img_3d
from .misc import slide_over_z, slide_over_ch

log = getLogger(__file__)


def get_elastix_log_dir():
    basedir = "./"
    names = os.listdir(basedir)
    ix = 0
    name = lambda: f".elastix.log.{ix}"
    while name() in names: ix += 1
    os.mkdir(name())
    return name()


class Registration(abc.ABC):
    def __init__(self,
                 cycles: t.List[np.ndarray],
                 ref_cycle: int = -1,
                 ref_channel: t.Union[int, str] = 'mean',
                 ref_z: t.Optional[t.Union[int, str]] = 'mean',
                 ref_gaussian_sigma: t.Optional[float] = None,
                ):
        self.cycles = cycles
        self.ref_cycle = ref_cycle
        self.ref_channel = ref_channel
        if ref_z is None:
            self.dim = 3
            self.ref_z = None
        else:
            self.dim = 2
            self.ref_z = ref_z
        self.ref_gaussian_sigma = ref_gaussian_sigma
        self.transforms = None

    def fetch_specified(self, im4d: np.ndarray) -> np.ndarray:
        if self.dim == 3:
            im = get_img_3d(im4d, self.ref_channel)
        else:
            im = get_img_2d(im4d, self.ref_channel, self.ref_z)
        if not (self.ref_gaussian_sigma is None):
            im = ndi.gaussian_filter(im, self.ref_gaussian_sigma)
        return im

    @abc.abstractmethod
    def estimate_transform(self):
        pass

    @abc.abstractmethod
    def apply(self):
        pass


class SitkBasedRegistration(Registration):

    def __init__(self,
                 cycles: t.List[np.ndarray],
                 ref_cycle: int = -1,
                 ref_channel: t.Union[int, str] = 'mean',
                 ref_z: t.Optional[t.Union[int, str]] = 'mean',
                 ref_gaussian_sigma: t.Optional[float] = None,
                 elastix_parameter_map='affine'
                 ):
        super().__init__(cycles, ref_cycle, ref_channel, ref_z, ref_gaussian_sigma)
        self.selx = sitk.ElastixImageFilter()
        try:
            pm = sitk.GetDefaultParameterMap(elastix_parameter_map)
        except RuntimeError:
            pm = sitk.ReadParameterFile(elastix_parameter_map)
        self.selx.SetParameterMap(pm)
        self.selx.LogToFileOn()
        elastix_log_dir = get_elastix_log_dir()
        log.info(f"Redirect elastix logs to '{elastix_log_dir}'")
        self.selx.SetOutputDirectory(elastix_log_dir)
        self.selx.LogToConsoleOff()

    def estimate_transform(self):
        ix_ref = self.ref_cycle
        selx = self.selx
        ref_im4d = self.cycles[ix_ref]
        ref_im = self.fetch_specified(ref_im4d)
        ref_itk = sitk.GetImageFromArray(ref_im)
        selx.SetFixedImage(ref_itk)
        transforms = []
        for idx, im4d in enumerate(self.cycles):
            if (idx == ix_ref) or (idx == (len(self.cycles) + ix_ref)):
                trans = None
            else:
                im = self.fetch_specified(im4d)
                im_itk = sitk.GetImageFromArray(im)
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
                def proc_im(im, *args):
                    im_itk = sitk.GetImageFromArray(im)
                    im_res = sitk.Transformix(im_itk, trans)
                    return sitk.GetArrayFromImage(im_res)
                if self.dim == 2:
                    res = slide_over_z(im4d, proc_im, 1)
                else:
                    res = slide_over_ch(im4d, proc_im, 1)
            aligned.append(res)
        return aligned


class MoveZ(Registration):
    def __init__(self,
                 cycles: t.List[np.ndarray],
                 ref_cycle: int = -1,
                 ref_channel: t.Union[int, str] = 'mean',
                 ref_gaussian_sigma: t.Optional[float] = None,
                 ):
        for cy in cycles:  # all image should be volumetric(more than one z layer)
            assert cy.shape[2] > 1
        super().__init__(cycles, ref_cycle, ref_channel, None, ref_gaussian_sigma)

    def estimate_transform(self):
        ix_ref = self.ref_cycle
        centers = []
        for im4d in self.cycles:
            im3d = self.fetch_specified(im4d)
            mean_along_z = im3d.mean(axis=0)
        self.transforms = centers

    def apply(self) -> t.List[np.ndarray]:
        pass
