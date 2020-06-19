from .base import ChainTool, SpotsIO, GenesIO
from ..lib.log import print_arguments
from ..lib.spots.decode import DistGraphDecode
from ..lib.barcode import read_codebook, get_code2chidxs

import logging


log = logging.getLogger(__file__)


class Decode(ChainTool, SpotsIO, GenesIO):
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
        self.dc = None
        self.coordinates = None
        self.dists_per_gene = None
        self.chidxs_per_gene = None

    def parse(self,
              code_book: str,
              channels: str = "AGCT",
              chars_per_cycle: int = 2,
              ):
        """Load codebook"""
        print_arguments(log.info)
        self.code2gene = cb = read_codebook(code_book)
        self.code2chidxs = get_code2chidxs(cb.keys(), channels, chars_per_cycle)
        return self

    def _check_codes_and_spots(self):
        if not self.spots:
            raise ValueError("please read spots firstly.")
        if not self.code2gene:
            raise ValueError("please parse barcodes firstly.")
        for code in self.code2gene.keys():
            gene = self.code2gene[code]
            chidxs = self.code2chidxs[code]
            try:
                for ixcy, ixch in enumerate(chidxs):
                    pts = self.spots[ixcy][ixch]
            except IndexError:
                msg = "Uncorrected cycle or channel indexing, " +\
                      f"when index gene {gene}, channel indexes: {chidxs}. " +\
                      f"spots shape: {[len(chs) for chs in self.spots]}"
                log.error(msg)
                raise ValueError(msg)

    def dist_graph(self, d: float):
        """Decode via distance graph"""
        print_arguments(log.info)
        self._check_codes_and_spots()
        self.dc = DistGraphDecode(self.spots)
        self.coordinates = []
        self.dists_per_gene = []
        self.chidxs_per_gene = []
        for code in self.code2gene.keys():
            gene = self.code2gene[code]
            log.debug(f"Decoding gene {gene}")
            chidxs = self.code2chidxs[code]
            pts, d_sum = self.dc.decode(chidxs, d)
            self.coordinates.append(pts)
            self.dists_per_gene.append(d_sum)
            self.chidxs_per_gene.append(chidxs)
        return self

    read = SpotsIO.read_spots
    write = GenesIO.write_genes
    count = GenesIO.count_genes