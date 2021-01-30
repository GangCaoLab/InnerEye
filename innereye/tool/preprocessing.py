from collections import Iterable
import typing as t
from ..lib.log import print_arguments
from ..lib.misc import local_arguments
from .base import ChainTool, ImgIO, Resetable
from logging import getLogger
from ..lib.img.transform import scale_to_255, bright_range_transform

log = getLogger(__file__)


class PreProcessing(ChainTool, ImgIO, Resetable):
    """Image pre-processing"""

    def __init__(self):
        self.cycles = None
        Resetable.__init__(self, "cycles")

    def print_shape(self):
        print_arguments(log.info)
        msg = ""
        for ixcy, cy in enumerate(self.cycles):
            shape = cy.shape
            msg += f"cycle: {ixcy}, shape: {shape}"
        log.info(msg)
        return self

    def crop(self,
             x: t.Optional[t.Tuple] = None,
             y: t.Optional[t.Tuple] = None,
             z: t.Optional[t.Tuple] = None,
             ch: t.Optional[t.Tuple] = None,
             cy: t.Optional[t.Tuple] = None,
             fixcy: t.Optional[t.Union[int, t.List[int]]] = None):
        """Crop image, pass-in tuple object represent the slice along
        this axis.

        Slice tuple is in format:
            (start, end) or (start, end, step) or None
        It will be converted to a slice object using `slice` function.
        If pass a `None` object, it equal to `:`(slice all).

        :param x: Slice tuple for x-axis.
        :param y: Slice tuple for y-axis.
        :param z: Slice tuple for z-axis.
        :param ch: Slice tuple for channel-axis.
        :param cy: Slice tuple for cycle-axis.
        :param fixcy: Only process specified cycles.
        :return:
        """
        print_arguments(log.info)
        sx, sy, sz, sch, scy = [
            slice(*i) if isinstance(i, Iterable) else slice(i)
            for i in (x, y, z, ch, cy)]
        cycles = []
        if not isinstance(fixcy, list):
            fixcy = [fixcy]
        for ixcy, img in enumerate(self.cycles):
            if (fixcy is None) or (ixcy in fixcy):
                img_ = img[sy, sx, sz, sch]
            else:
                img_ = img
            cycles.append(img_)
        cycles = cycles[scy]
        self.set_new(cycles)
        return self

    def select(self,
               exclude: bool = False,
               x: t.Optional[t.List[int]] = None,
               y: t.Optional[t.List[int]] = None,
               z: t.Optional[t.List[int]] = None,
               ch: t.Optional[t.List[int]] = None,
               cy: t.Optional[t.List[int]] = None):
        """Select or exclude images in specified dimension.

        :param exclude: If true will exclude specified images, else keep them.
        :param x: Indexes for x-axis.
        :param y: Indexes for y-axis.
        :param z: Indexes for z-axis.
        :param ch: Indexes for channel-axis.
        :param cy: Indexes for cycle-axis.
        :return:
        """
        print_arguments(log.info)
        selectors = [x, y, z, ch, cy]
        new_cycles = []
        for cy_idx, img in enumerate(self.cycles):
            slices = [i if isinstance(i, list) else slice(None) for i in selectors]
            if exclude:
                for idx in range(len(slices)):
                    if selectors[idx] is None:
                        continue
                    len_all = len(self.cycles) if idx == 4 else img.shape[idx]
                    slices[idx] = [i for i in range(len_all) if i not in selectors[idx]]
            new_img = img[slices[0], :, :, :]
            new_img = new_img[:, slices[1], :, :]
            new_img = new_img[:, :, slices[2], :]
            new_img = new_img[:, :, :, slices[3]]
            if (cy is None) or (cy_idx in slices[-1]):
                new_cycles.append(new_img)
        self.set_new(new_cycles)
        return self

    def resize_z(self, to="max"):
        """Resize all cycles to same size in z-axis"""
        print_arguments(log.info)
        from skimage.transform import resize

        sorted_cys = sorted(self.cycles, key=lambda im: im.shape[2])
        if to == "max":
            target = sorted_cys[-1]
        elif to == "median":
            target = sorted_cys[len(sorted_cys)//2]
        else:
            target = sorted_cys[0]
        target_size = target.shape
        target_ix = self.cycles.index(target)
        cycles = []
        for ixcy, cy in enumerate(self.cycles):
            if ixcy == target_ix:
                cycles.append(cy)
            else:
                cycles.append(resize(cy, target_size))
        self.set_new(cycles)
        return self

    def z_stack(self):
        """Stack along z-axis."""
        print_arguments(log.info)
        cycles = [arr.mean(axis=2, keepdims=True) for arr in self.cycles]
        self.set_new(cycles)
        return self

    def adjust_gamma(self, gamma=1.5):
        """Perform gamma-adjust on image."""
        from skimage.exposure import adjust_gamma
        print_arguments(log.info)
        cycles = [adjust_gamma(arr, gamma) for arr in self.cycles]
        self.set_new(cycles)
        return self

    def registration(self,
                     ref_cycle=-1,
                     ref_channel='mean',
                     ref_z='mean',
                     ref_gaussian_sigma=None,
                     elastix_parameter_map='affine'):
        """Image registration, align images to the reference cycle."""
        from ..lib.img.registration import SitkBasedRegistration
        print_arguments(log.info)
        args = local_arguments(keywords=False)
        reg = SitkBasedRegistration(self.cycles, *args)
        reg.estimate_transform()
        aligned = reg.apply()
        self.set_new(aligned)
        return self

    def scale_to_255(self):
        print_arguments(log.info)
        cycles = [scale_to_255(c) for c in self.cycles]
        self.set_new(cycles)
        return self

    def bright_range_transform(self, bright_range):
        print_arguments(log.info)
        import numpy as np
        if isinstance(bright_range, tuple):
            cycles = [bright_range_transform(c, bright_range) for c in self.cycles]
        elif isinstance(bright_range, list):
            cycles = []
            for im4d in self.cycles:
                assert len(bright_range) == im4d.shape[-1]
                im3d_chs = []
                for ixch in range(im4d.shape[-1]):
                    im3d_ch = im4d[:,:,:,ixch]
                    rg = bright_range[ixch]
                    if rg is not None:
                        im3d_ch = bright_range_transform(im3d_ch, rg)
                    im3d_chs.append(im3d_ch)
                im4d_t = np.stack(im3d_chs, -1)
                cycles.append(im4d_t)
        else:
            raise ValueError("bright_range should be tuple or list of tuples")
        self.set_new(cycles)
        return self

    def gaussian_filter(self, sigma):
        print_arguments(log.info)
        import scipy.ndimage as ndi
        import numpy as np
        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = [sigma for _ in range(self.cycles[0].shape[-1])]
        assert isinstance(sigma, list)
        cycles = []
        for im4d in self.cycles:
            assert len(sigma) == im4d.shape[-1]
            im3d_chs = []
            for ixch in range(im4d.shape[-1]):
                im3d_ch = im4d[:,:,:,ixch]
                sm = sigma[ixch]
                if sm is not None:
                    im3d_ch = ndi.gaussian_filter(im3d_ch, sm)
                im3d_chs.append(im3d_ch)
            im4d_t = np.stack(im3d_chs, -1)
            cycles.append(im4d_t)
        self.set_new(cycles)
        return self

    def DoG(self, kw_dog={"low_sigma": 1, "high_sigma": 4}):
        print_arguments(log.info)
        import numpy as np
        from skimage.filters import difference_of_gaussians
        if isinstance(kw_dog, dict):
            kw_dog = [kw_dog for _ in range(self.cycles[0].shape[-1])]
        cycles = []
        for im4d in self.cycles:
            assert len(kw_dog) == im4d.shape[-1]
            im3d_chs = []
            for ixch in range(im4d.shape[-1]):
                im3d_ch = im4d[:,:,:,ixch]
                kw = kw_dog[ixch]
                if kw is not None:
                    im3d_ch = difference_of_gaussians(im3d_ch, **kw)
                im3d_chs.append(im3d_ch)
            im4d_t = np.stack(im3d_chs, -1)
            cycles.append(im4d_t)
        self.set_new(cycles)
        return self

    read = ImgIO.read_img
    write = ImgIO.write_img
