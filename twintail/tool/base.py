from twintail.utils.log import print_arguments
from twintail.utils.io.h5 import read_cycles, write_cycles
from twintail.utils.io.h5 import read_spots, write_spots

import numpy as np

from logging import getLogger
log = getLogger(__file__)


class ChainTool(object):
    """Chaining tools base class."""
    def end(self):
        """Ending process"""
        log.info(f"Ending {self.__class__.__name__}.")

    def read(self, path: str):
        """Load images to memory."""
        print_arguments(log.info)
        self.cycles = read_cycles(path)
        self.dimensions = [img.shape for img in self.cycles]
        return self

    def write(self, path: str):
        """Write back images to disk."""
        print_arguments(log.info)
        write_cycles(path, self.cycles)
        return self

    def clear(self):
        """Clear memory"""
        print_arguments(log.info)
        del self.cycles
        return self


class SpotsTool(ChainTool):

    def read(self, path: str):
        """Read spot coordinates from disk"""
        print_arguments(log.info)
        spots, dims = read_spots(path)
        self.spots = spots
        self.dimensions = dims
        return self

    def write(self, path: str):
        """Write spot coordinates to disk"""
        print_arguments(log.info)
        dims = [dim[:3] for dim in self.dimensions]
        write_spots(path, self.spots, dims)
        return self

    def clear(self):
        """Clear memory"""
        print_arguments(log.info)
        del self.spots
        return self

    def count(self, outfile=None, z=True):
        """Count number of points in each cycle and channel.

        :param outfile: Write count result to specified file.
        :param z: Count each z layer or not.
        :return:
        """
        print_arguments(log.info)
        msg = ""
        for ixcy, chs in enumerate(self.spots):
            msg += f"Cycle index: {ixcy}\n"
            for ixch, coords in enumerate(chs):
                msg += f"\tChannel index: {ixch}\n"
                if z:
                    for z in np.unique(coords[:, 2]):
                        layer = coords[coords[:, 2] == z]
                        msg += f"\t\t{z}\t{layer.shape[0]}\n"
                else:
                    msg += f"\t\t{coords.shape[0]}\n"
        log.info(msg)
        if outfile:
            with open(outfile, 'w') as f:
                f.write(msg)
        return self

