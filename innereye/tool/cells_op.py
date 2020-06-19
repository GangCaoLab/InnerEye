import numpy as np

from .base import ChainTool, ImgIO, GenesIO, CellsIO
from ..lib.img.misc import slide_over_z, get_img_3d
from ..lib.misc import local_arguments
from ..lib.log import print_arguments
from ..lib.img.cells import otsu_watershed_2d, gene2cell_2d


from logging import getLogger
log = getLogger(__file__)


class CellsOp(ChainTool, ImgIO, GenesIO, CellsIO):

    read = CellsIO.read_cells
    write = CellsIO.write_cells
    count = CellsIO.count_cells

    def __init__(self,
                 ref_cycle: int = 0,
                 channel: str = 'mean',
                 z_mode: str = 'slide',
                 n_workers: int = 1):
        self.ref_cycle = ref_cycle
        self.channel = channel
        self.z_mode = z_mode
        self.n_workers = n_workers
        self.cycles = None
        self.cells_center = None
        self.cells_mask = None
        self.cell_assign = None
        self.code2gene = None
        self.coordinates = None

    def call_cells(self,
                   gaussian_sigma: int = 8,
                   min_cc_size: int = 500,
                   merge_radius: int = 10,
                   ):
        """Calculate all cell's center and mask."""
        print_arguments(log.info)
        img4d = self.cycles[self.ref_cycle]
        img3d = get_img_3d(img4d, self.channel)
        img4d_ = img3d[:, :, :, np.newaxis]
        args = local_arguments(keywords=False)
        if self.z_mode == 'slide':
            f = lambda im2d, _: otsu_watershed_2d(im2d, *args)
            chs = slide_over_z(img4d_, f, self.n_workers, stack_z=False, stack_ch=False)
            zs = chs[0]
            masks_, centers_ = [], []
            for z, (cts, mask) in enumerate(zs):
                z_ = np.full((cts.shape[0], 1), z)
                centers_.append(np.c_[cts, z_])
                masks_.append(mask)
            self.cells_center = np.concatenate(centers_)
            self.cells_mask = np.stack(masks_, -1)
        else:
            raise NotImplementedError
        return self

    def gene2cell(self, dist_th=50):
        """Assign gene to cell."""
        print_arguments(log.info)
        if (self.code2gene is None) or (self.coordinates is None):
            raise IOError("Should read gene firstly.")
        img3d = self.cells_mask
        img4d_ = img3d[:, :, :, np.newaxis]
        centers = self.cells_center
        gene_pos = self.coordinates
        if self.z_mode == 'slide':
            def func(im2d, idx):
                _, z = idx
                cts = centers[centers[:, 2] == z, :2]
                pos = [gp[gp[:, 2] == z, :2] for gp in gene_pos]
                return [gene2cell_2d(cts, im2d, gp, dist_th)
                        if gp.shape[0] > 0 else np.zeros((0, 2), cts.dtype)
                        for gp in pos]
            chs = slide_over_z(img4d_, func, self.n_workers, stack_ch=False, stack_z=False)
            zs = chs[0]
            pos_per_gene = []
            for z in range(len(zs)):
                for i, zsg in enumerate(zs[z]):
                    if len(pos_per_gene) <= i:
                        pos_per_gene.append(zsg)
                    else:
                        old = pos_per_gene[i]
                        pos_per_gene[i] = np.concatenate([old, zsg])
            self.cell_assign = pos_per_gene
        else:
            raise NotImplementedError
        return self
