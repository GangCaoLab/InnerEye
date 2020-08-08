import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
import random
import typing as t

from .base import ChainTool, ImgIO, SpotsIO, GenesIO, CellsIO
from ..lib.img.misc import get_img_2d
from ..lib.log import print_arguments

from logging import getLogger
log = getLogger(__file__)


def marker_styles(cmap="hsv", seed=0):
    markers = [".", "v", "^", "<", ">",
               "1", "2", "3", "4", "s",
               "p", "P", "*", "+", "x"]
    ix = 0
    cmap_ = cm.get_cmap(cmap)
    random.seed(seed)
    while True:
        yield cmap_(random.random()), markers[ix % len(markers)]
        ix += 1


@dataclass
class Img2dConf:
    img_path: str
    cycle: t.List[int]
    channel: t.List[int]
    z: t.List[int]
    show_cells_mask: bool = True
    show_cells_center: bool = True
    show_genes: bool = True
    show_spots: bool = True


def _compose_list(x, elm_tp=int, mean_alt=None):
    if type(x) is elm_tp:
        res = [x]
    elif type(x) is list:
        res = x
    elif x == 'mean':
        res = mean_alt
    else:
        raise TypeError(f"{x} is not valid type.")
    return res


class Plot2d(ChainTool, ImgIO, SpotsIO, GenesIO, CellsIO):

    def __init__(self):
        self.img = None
        self.spots = None
        self.dimensions = None
        self.code2gene = None
        self.coordinates = None
        self.cells_mask = None
        self.cells_center = None
        self.figsize = None
        self.cell_assign = None
        self.img_path = None
        self.imgs = []
        self.confs: t.List[Img2dConf] = []

    def read_img(self, path: str):
        self.img_path = path
        return super().read_img(path)

    def background(self, cycle=0, channel='mean', z='mean',
                   show_cells_mask=True, show_cells_center=True,
                   show_gene=True, show_spots=True):
        print_arguments(log.info)
        assert type(cycle) is int
        assert (type(channel) is int) or (channel == 'mean') or (type(channel) is list)
        assert (type(z) is int) or (z == 'mean') or (type(z) is list)
        im4d = self.cycles[cycle]
        self.img = get_img_2d(im4d, channel, z)
        self.imgs.append(self.img)
        _channels_select = _compose_list(channel, int, list(range(im4d.shape[3])))
        _z_select = _compose_list(z, int, list(range(im4d.shape[2])))
        conf = Img2dConf(self.img_path, [cycle], _channels_select, _z_select,
                         show_cells_mask, show_cells_center,
                         show_gene, show_spots)
        self.confs.append(conf)
        return self

    def plot(self, figpath=None, figsize=(10, 10), n_cols_max=2, legend_path=None):
        print_arguments(log.info)
        if figpath:
            fig = self._draw(figsize, n_cols_max, legend_path)
            fig.tight_layout()
            fig.savefig(figpath)
        else:
            from ..lib.ui import MainWindow
            from PyQt5 import QtWidgets
            app = QtWidgets.QApplication([])
            w = MainWindow(self, figsize, n_cols_max, legend_path)
            app.exec()
        return self

    def _draw(self, figsize, n_cols_max, legend_path):
        self.figsize = figsize
        fig = plt.figure(figsize=figsize)
        n = len(self.confs)
        nrows = 1 + ((n-1) // n_cols_max)
        ncols = min(n, n_cols_max)
        gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, wspace=0.2, hspace=0.2)
        _l_path = os.path.splitext(legend_path) if legend_path else None
        axes = []
        for i in range(n):
            if _l_path:
                l_path = _l_path[0] + f".{i}" + _l_path[1] if n > 1 else legend_path
            else:
                l_path = None
            y = i // n_cols_max
            x = i % n_cols_max
            if i > 0:
                ax = fig.add_subplot(gs[y, x], sharex=axes[0], sharey=axes[0])
            else:
                ax = fig.add_subplot(gs[y, x])
            axes.append(ax)
            self._draw_one(ax, self.imgs[i], self.confs[i], l_path)
        return fig

    def _draw_one(self, ax, img, conf: Img2dConf, legend_path):
        if img is not None:
            ax.imshow(img, cmap='gray')
        if conf.show_cells_mask and (self.cells_mask is not None):
            self._plot_cells_mask(ax)
        if conf.show_cells_center and (self.cells_center is not None):
            self._plot_cells_center(ax)
        if conf.show_genes and (self.code2gene is not None):
            shapes, labels = self._plot_genes(ax)
            self._plot_legend(ax, legend_path, shapes, labels)
            if self.cell_assign is not None:
                self._plot_assign(ax)
        elif conf.show_spots and (self.spots is not None):
            shapes, labels = self._plot_spots(ax, conf.z, conf.channel)
            self._plot_legend(ax, legend_path, shapes, labels)
        ax.set_ylim(img.shape[0], 0)
        ax.set_xlim(0, img.shape[1])
        title = f"cycle: {conf.cycle}, channel: {conf.channel}, z: {conf.z}"
        ax.set_title(title)

    def _plot_legend(self, ax, legend_path, shapes, labels):
        if legend_path:
            n = len(shapes)
            fig_legend, ax_legend = plt.subplots(figsize=(3, max(4.0, 0.25 * n)))
            ax_legend.legend(shapes, labels, loc=9)
            ax_legend.axis('off')
            fig_legend.tight_layout()
            fig_legend.savefig(legend_path)
        else:
            ax.legend(framealpha=0.5)

    def _estimate_scatter_size(self):
        s = 5 * (self.figsize[0] * self.figsize[1]) // 100
        return s

    def _plot_genes(self, ax):
        marker_gen = marker_styles()
        s = self._estimate_scatter_size()
        shapes = []; labels = []
        for ix, gene in enumerate(self.code2gene.values()):
            c, m = next(marker_gen)
            pts = self.coordinates[ix][:, :2]
            if pts.shape[0] == 0:
                continue
            sh = ax.scatter(pts[:, 1], pts[:, 0],
                            c=[c for _ in range(pts.shape[0])],
                            marker=m, s=s,
                            label=gene)
            shapes.append(sh)
            labels.append(gene)
        return shapes, labels

    def _plot_cells_mask(self, ax, z=0):
        assert (type(z) is int) or (z == 'mean')
        if z == 'mean':
            im_ = self.cells_mask.mean(axis=2)
        else:
            im_ = self.cells_mask[:, :, 0]
        im_ = im_.astype(np.float32)
        im_[im_ < 1] = np.nan
        ax.imshow(im_, interpolation='none', vmin=0, alpha=0.3, cmap='prism')

    def _plot_cells_center(self, ax, z=0):
        centers = self.cells_center
        pos = centers[centers[:, 2] == z, :2]
        s = 15 * (self.figsize[0] * self.figsize[1]) // 100
        ax.scatter(pos[:, 1], pos[:, 0],
                   c=['red' for _ in range(pos.shape[0])],
                   marker='o',
                   s=s,
                   )

    def _plot_assign(self, ax):
        for ix, _ in enumerate(self.code2gene.values()):
            assign = self.cell_assign[ix]
            pts = self.coordinates[ix][:, :2]
            for i in range(pts.shape[0]):
                from_ = pts[i]
                to_ = assign[i]
                if np.all(~np.isnan(to_)):
                    ax.plot([from_[1], to_[1]], [from_[0], to_[0]],
                            linewidth=0.5,
                            color='red',
                            alpha=0.6,)

    def _plot_spots(self, ax, z_select, channels_select):
        marker_gen = marker_styles()
        s = self._estimate_scatter_size()
        shapes = []; labels = []
        for ixcy, channels in enumerate(self.spots):
            for ixch, coords in enumerate(channels):
                c, m = next(marker_gen)
                pts = coords[np.where(np.isin(coords[:, 2], z_select))[0]]
                pts = pts[:, :2]
                if pts.shape[0] == 0:
                    continue
                if ixch not in channels_select:
                    continue
                label = f"cycle_ix: {ixcy}, channel_ix: {ixch}"
                sh = ax.scatter(pts[:, 1], pts[:, 0],
                                c=[c for _ in range(pts.shape[0])],
                                marker=m, s=s, alpha=0.5,
                                label=label)
                shapes.append(sh)
                labels.append(label)
        return shapes, labels

