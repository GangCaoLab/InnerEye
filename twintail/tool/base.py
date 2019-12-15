from twintail.utils.log import print_arguments
from twintail.utils.io.h5 import read_cycles, write_cycles
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
        return self

    def write(self, path: str):
        """Write back images to disk."""
        print_arguments(log.info)
        write_cycles(path, self.cycles)
        return self

