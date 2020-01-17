from .base import SpotsTool
from twintail.utils.log import print_arguments

import logging

from ..utils.barcode import read_codebook, get_code2chidxs

log = logging.getLogger(__file__)


class Decode(SpotsTool):
    """Decode signal from spots."""
    def __init__(self,
                 z_mode: str = 'slide',
                 n_workers: int = 1
                 ):
        print_arguments(log.info)
        assert z_mode in {'slide', 'whole'}
        self.z_mode = z_mode
        self.n_workers = n_workers
        self.spots = None
        self.dimensions = None
        self.code2gene = None
        self.code2chidxs = None

    def parse(self,
              code_book: str,
              channels: str = "AGCT",
              chars_per_cycle: int = 2,
              ):
        """Load codebook"""
        print_arguments(log.info)
        self.code2gene = cb = read_codebook(code_book)
        self.code2chidxs = get_code2chidxs(cb.keys(), channels, chars_per_cycle)

    def dist_graph(self, d: float):
        print_arguments(log.info)
        for code in self.code2gene.keys():
            gene = self.code2gene[code]
            chidxs = self.code2chidxs[code]


