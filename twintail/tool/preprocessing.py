from twintail.utils.io.h5 import read_cycles, write_cycles
from twintail.utils.log import print_arguments
from collections import Iterable

from logging import getLogger
log = getLogger(__file__)


class PreProcessing(object):
    """Image pre-processing"""

    def __init__(self):
        self.cycles = None

    def read(self, path: str):
        """Load images to memory."""
        print_arguments(log.info)
        self.cycles = read_cycles(path)
        return self

    def write(self, path: str):
        """Write back to disk."""
        print_arguments(log.info)
        write_cycles(path, self.cycles)

    def crop(self, x=None, y=None, z=None, ch=None, cy=None):
        """Crop image."""
        print_arguments(log.info)
        sx, sy, sz, sch, scy = [
            slice(*i) if isinstance(i, Iterable) else slice(i)
            for i in (x, y, z, ch, cy)]
        self.cycles = [img[sx, sy, sz, sch] for img in self.cycles][scy]
        return self

    def z_stack(self):
        """Stack along z-axis."""
        print_arguments(log.info)
        self.cycles = [arr.mean(axis=2) for arr in self.cycles]
        return self

    def adjust_gamma(self, gamma=1.5):
        """Perform gamma-adjust on image."""
        from skimage.exposure import adjust_gamma
        print_arguments(log.info)
        self.cycles = [adjust_gamma(arr, gamma) for arr in self.cycles]
        return self

