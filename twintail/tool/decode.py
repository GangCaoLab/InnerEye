from .base import ChainTool
from twintail.utils.io.h5 import read_spots
from twintail.utils.log import print_arguments

import logging
log = logging.getLogger(__file__)


class Decode(ChainTool):
    """Decode signal from spots."""
    def __init__(self,
                 z_mode: str = 'slide',
                 n_workers: int = 1):
        print_arguments(log.info)
        assert z_mode in {'slide', 'whole'}
        self.z_mode = z_mode
        self.n_workers = n_workers
        self.cycles = None
        self.dimensions = None

    def read(self, path: str):
        """Load coordinates of spots into memory."""
        print_arguments(log.info)
        self.cycles, self.dimensions = read_spots(path)
        return self

