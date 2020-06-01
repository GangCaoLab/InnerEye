from collections import Iterable
import typing as t
from ..lib.log import print_arguments
from ..lib.misc import local_arguments
from .base import ChainTool, ImgIO
from logging import getLogger

log = getLogger(__file__)


class PreProcessing(ChainTool, ImgIO):
    """Image pre-processing"""

    def __init__(self):
        self.cycles = None

    def crop(self,
             x: t.Optional[t.Tuple] = None,
             y: t.Optional[t.Tuple] = None,
             z: t.Optional[t.Tuple] = None,
             ch: t.Optional[t.Tuple] = None,
             cy: t.Optional[t.Tuple] = None):
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
        :return:
        """
        print_arguments(log.info)
        sx, sy, sz, sch, scy = [
            slice(*i) if isinstance(i, Iterable) else slice(i)
            for i in (x, y, z, ch, cy)]
        self.cycles = [img[sy, sx, sz, sch] for img in self.cycles][scy]
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
            new_img = img[slices[0], slices[1], slices[2], slices[3]]
            if cy and cy_idx in slices[-1]:
                new_cycles.append(new_img)
        self.cycles = new_cycles
        return self

    def z_stack(self):
        """Stack along z-axis."""
        print_arguments(log.info)
        self.cycles = [arr.mean(axis=2, keepdims=True) for arr in self.cycles]
        return self

    def adjust_gamma(self, gamma=1.5):
        """Perform gamma-adjust on image."""
        from skimage.exposure import adjust_gamma
        print_arguments(log.info)
        self.cycles = [adjust_gamma(arr, gamma) for arr in self.cycles]
        return self

    def registration(self,
                     ref_cycle=-1, ref_channel='mean',
                     ref_z='mean',
                     elastix_parameter_map='affine'):
        """Image registration, align images to the reference cycle."""
        from ..lib.img.registration import Registration2d
        print_arguments(log.info)
        args = local_arguments(keywords=False)
        reg = Registration2d(self.cycles, *args)
        reg.estimate_transform()
        aligned = reg.apply()
        self.cycles = aligned
        return self

    read = ImgIO.read_img
    write = ImgIO.write_img
