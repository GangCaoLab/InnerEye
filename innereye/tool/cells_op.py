import numpy as np
import pandas as pd

from .base import ChainTool, ImgIO, GenesIO, CellsIO
from ..lib.img.misc import slide_over_z, get_img_3d
from ..lib.misc import local_arguments
from ..lib.log import print_arguments
from ..lib.img.cells import otsu_watershed_2d, otsu_cc_center_2d, gene2cell_2d


from logging import getLogger
log = getLogger(__file__)


def func_for_slide_call_cell(func, args, img4d, channel, z_mode, n_workers):
    img3d = get_img_3d(img4d, channel)
    img4d_ = img3d[:, :, :, np.newaxis]
    if z_mode == 'slide':
        f = lambda im2d, _: func(im2d, *args)
        chs = slide_over_z(img4d_, f, n_workers, stack_z=False, stack_ch=False)
        zs = chs[0]
        masks_, centers_ = [], []
        for z, (cts, mask) in enumerate(zs):
            z_ = np.full((cts.shape[0], 1), z)
            centers_.append(np.c_[cts, z_])
            masks_.append(mask)
        return np.concatenate(centers_), np.stack(masks_, -1)
    else:
        raise NotImplementedError


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

    def call_with_dist_watershed(self,
            gaussian_sigma: int = 8,
            min_cc_size: int = 500,
            merge_radius: int = 10,
        ):
        """Calculate all cell's center and mask via dist_watershed method."""
        print_arguments(log.info)
        args = local_arguments(keywords=False)
        im4d = self.cycles[self.ref_cycle]
        self.cells_center, self.cells_mask = func_for_slide_call_cell(
            otsu_watershed_2d, args,
            im4d, self.channel, self.z_mode, self.n_workers
        )
        return self

    def call_with_cc_center(self,
            gaussian_sigma: int = 8,
            min_cc_size: int = 500,
        ):
        """Calculate all cell's center and mask via cc_center method."""
        print_arguments(log.info)
        args = local_arguments(keywords=False)
        im4d = self.cycles[self.ref_cycle]
        self.cells_center, self.cells_mask = func_for_slide_call_cell(
            otsu_cc_center_2d, args,
            im4d, self.channel, self.z_mode, self.n_workers
        )
        return self

    def gene2cell(self,
            dist_th=50,
            max_iter=20,
            iter_dist_step=20,
        ):
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
                return [gene2cell_2d(cts, im2d, gp, dist_th, max_iter, iter_dist_step)
                        if gp.shape[0] > 0 else np.zeros((0, 2), cts.dtype)
                        for gp in pos]
            chs = slide_over_z(img4d_, func, self.n_workers, stack_ch=False, stack_z=False)
            zs = chs[0]
            pos_per_gene = []
            for z in range(len(zs)):
                for i, zsg in enumerate(zs[z]):
                    z_ = np.full((zsg.shape[0], 1), z)
                    pos = np.c_[zsg, z_]
                    if len(pos_per_gene) <= i:
                        pos_per_gene.append(pos)
                    else:
                        old = pos_per_gene[i]
                        pos_per_gene[i] = np.concatenate([old, pos])
            self.cell_assign = pos_per_gene
        else:
            raise NotImplementedError
        return self

    def cell_quanta(self, path):
        """Output cell quanta result table."""
        print_arguments(log.info)
        cells = {str(tuple([float(c[i]) for i in range(3)])): {} for c in self.cells_center}
        if self.z_mode == "slide":
            for g_idx, (_, gene) in enumerate(self.code2gene.items()):
                col = []
                assign = self.cell_assign[g_idx]
                centers, cnts = np.unique(assign, return_counts=True, axis=0)
                non_nan_idxs = ~np.isnan(centers[:,0])
                centers = centers[non_nan_idxs]
                cnts = cnts[non_nan_idxs]
                for i, c in enumerate(centers):
                    cell = str(tuple([float(c[i]) for i in range(3)]))
                    cells[cell][gene] = cnts[i]
        else:
            raise NotImplementedError
        df = pd.DataFrame(cells).T
        sep = "," if path.endswith(".csv") else "\t"
        df.to_csv(path, sep=sep)
        return self
