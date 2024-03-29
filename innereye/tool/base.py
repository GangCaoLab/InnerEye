from ..lib.log import print_arguments
from ..lib.io.h5 import read_cycles, write_cycles
from ..lib.io.h5 import read_spots, write_spots, read_decode, write_decode
from ..lib.io.h5 import read_cells, write_cells, read_assign, write_assign

import typing as t
from collections import OrderedDict as od
from collections import deque
import numpy as np

from logging import getLogger
log = getLogger(__file__)


class ChainTool(object):
    """Chaining tools base class."""
    def end(self):
        """Ending process"""
        log.info(f"Ending {self.__class__.__name__}.")


class ImgIO(object):
    def read_img(self, path: str):
        """Load images to memory."""
        print_arguments(log.info)
        self.cycles = read_cycles(path)
        self.dimensions = [img.shape for img in self.cycles]
        return self

    def write_img(self, path: str):
        """Write back images to disk."""
        print_arguments(log.info)
        write_cycles(path, self.cycles)
        return self

    def clear_img(self):
        """Clear memory"""
        print_arguments(log.info)
        del self.cycles
        return self


class MaskIO(object):
    def read_mask(self, path: str):
        """Load mask to memory."""
        print_arguments(log.info)
        self.masks = read_cycles(path)
        self.dimensions = [img.shape for img in self.masks]
        return self

    def write_mask(self, path: str):
        """Write back masks to disk."""
        print_arguments(log.info)
        write_cycles(path, self.masks)
        return self

    def clear_mask(self):
        """Clear memory"""
        print_arguments(log.info)
        del self.masks
        return self


def log_msg(msg, outfile=None):
    log.info(msg)
    if outfile:
        with open(outfile, 'w') as f:
            f.write(msg)


class SpotsIO(object):

    def read_spots(self, path: str):
        """Read spot coordinates from disk"""
        print_arguments(log.info)
        spots, dims = read_spots(path)
        self.spots = spots
        self.dimensions = dims
        return self

    def write_spots(self, path: str):
        """Write spot coordinates to disk"""
        print_arguments(log.info)
        dims = [dim[:3] for dim in self.dimensions]
        write_spots(path, self.spots, dims)
        return self

    def clear_spots(self):
        """Clear memory"""
        print_arguments(log.info)
        del self.spots
        return self

    def count_spots(self, outfile=None, show_z=True):
        """Count number of points in each cycle and channel.

        :param outfile: Write count result to specified file.
        :param show_z: Count each z layer or not.
        :return:
        """
        print_arguments(log.info)
        msg = ""
        for ixcy, chs in enumerate(self.spots):
            msg += f"Cycle index: {ixcy}\n"
            for ixch, coords in enumerate(chs):
                if hasattr(self, 'combs') and self.combs:
                    combs = getattr(self, 'combs')
                    msg += f"\tChannel index: {ixch}{tuple(combs[ixch])}\n"
                else:
                    msg += f"\tChannel index: {ixch}\n"
                if show_z:
                    if coords.shape[0] <= 0:
                        continue
                    for z in np.unique(coords[:, 2]):
                        layer = coords[coords[:, 2] == z]
                        msg += f"\t\tz: {int(z)}\tcount: {layer.shape[0]}\n"
                else:
                    msg += f"\t\t{coords.shape[0]}\n"
        log_msg(msg, outfile)
        return self


class GenesIO(object):
    """IO trait for decoded signals"""

    def read_genes(self, path: str):
        """Read decode result from disk"""
        print_arguments(log.info)
        (
            genes, points_per_gene, dists_per_gene,
            barcodes_per_gene, chidxs_per_gene,
            unmatch_coordinates, unmatch_dists, unmatch_chidxs
        ) = read_decode(path)
        self.code2gene = od(zip(barcodes_per_gene, genes))
        self.coordinates = points_per_gene
        self.dists_per_gene = dists_per_gene
        self.chidxs_per_gene = chidxs_per_gene
        self.coordinates_unmatch = unmatch_coordinates
        self.dists_per_unmatch = unmatch_dists
        self.chidxs_per_unmatch = unmatch_chidxs
        return self

    def write_genes(self, path: str):
        """Write decode result to disk."""
        print_arguments(log.info)
        write_decode(path,
                     list(self.code2gene.values()),
                     self.coordinates,
                     self.dists_per_gene,
                     list(self.code2gene.keys()),
                     self.chidxs_per_gene,
                     self.coordinates_unmatch,
                     self.dists_unmatch,
                     self.chidxs_unmatch,
                     )
        return self

    def count_genes(self, outfile=None):
        """Count decode result."""
        print_arguments(log.info)
        info = "Decode result count:\n"
        for ix, (code, gene) in enumerate(self.code2gene.items()):
            pts = self.coordinates[ix]
            info += f"{gene}\t{code}\t{pts.shape[0]}\n"
        log_msg(info, outfile)
        return self


class CellsIO(object):
    """IO trait for Cell positions."""

    def read_cells(self, path: str):
        """Read cell positions from disk."""
        print_arguments(log.info)
        center, mask = read_cells(path)
        self.cells_center = center
        self.cells_mask = mask
        return self

    def write_cells(self, path: str):
        """Write cells position to disk."""
        print_arguments(log.info)
        write_cells(path,
                    self.cells_center,
                    self.cells_mask,
                    )
        return self

    def count_cells(self, outfile=None, show_z=True):
        """Count number of cells.

        :param outfile: Write count result to specified file.
        :param show_z: Count each z layer or not.
        :return:
        """
        print_arguments(log.info)
        from ..lib.img.cells import cell_area_counts
        msg = "Count cells:\n"
        msg += "z\tcount\tarea_mean\tarea_std\n"
        centers = self.cells_center
        mask = self.cells_mask
        if show_z:
            for z in np.sort(np.unique(centers[:, 2])):
                z = int(z)
                in_z = centers[centers[:, 2] == z]
                im2d = mask[:, :, z]
                area_mean, area_std = cell_area_counts(im2d)
                msg += f"{z}\t{in_z.shape[0]}\t{area_mean}\t{area_std}\n"
        else:
            raise NotImplementedError
        log_msg(msg, outfile)
        return self

    def read_assign(self, path: str):
        """Read gene's cell assign from disk."""
        print_arguments(log.info)
        self.cell_assign = read_assign(path, list(self.code2gene.values()))
        return self

    def write_assign(self, path: str):
        """Write gene's cell assign to disk."""
        print_arguments(log.info)
        write_assign(path, list(self.code2gene.values()), self.cell_assign)
        return self


class Resetable(object):
    def __init__(self, attr_names: t.Union[str, t.List[str]], limit: int = 1):
        if not isinstance(attr_names, list):
            attr_names = [attr_names]
        self._attr_names = attr_names
        self._memory = {
            n: deque(maxlen=limit) for n in attr_names
        }
    
    def _check_name(self, name):
        if name is None:
            name = self._attr_names[0]
        if name not in self._attr_names:
            raise ValueError("name should in attr_names")
        return name

    def set_new(self, new, name: t.Optional[str] = None):
        name = self._check_name(name)
        old_attr = getattr(self, name)
        self._memory[name].append(old_attr)
        setattr(self, name, new)

    def reset(self, index: int = -1, name: t.Optional[str] = None):
        name = self._check_name(name)
        old_attr = self._memory[name][index]
        setattr(self, name, old_attr)
        return self

    def clear_old(self, name: t.Optional[str] = None):
        self._check_name(name)
        names = self._attr_names if name is None else [name]
        for n in names:
            self._memory[n].clear()
        return self

