from .base import ChainTool, ImgIO, GenesIO, CellsIO
from ..lib.img.misc import get_img_2d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random


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


class Plot2d(ChainTool, ImgIO, GenesIO, CellsIO):

    def __init__(self, show_center=True):
        self.show_center = show_center
        self.img = None
        self.code2gene = None
        self.coordinates = None
        self.cells_mask = None
        self.cells_center = None
        self.figsize = None
        self.cell_assign = None

    def background(self, cycle=0, channel='mean', z='mean'):
        assert type(cycle) is int
        assert (type(channel) is int) or (channel == 'mean')
        assert (type(z) is int) or (z == 'mean')
        im4d = self.cycles[cycle]
        self.img = get_img_2d(im4d, channel, z)
        return self

    def plot(self, figpath=None, figsize=(10, 10), legend_path="./legend.png"):
        self.figsize = figsize
        fig, ax = plt.subplots(figsize=figsize)
        if self.img is not None:
            ax.imshow(self.img, cmap='gray')
        if self.cells_mask is not None:
            self._plot_cells_mask(ax)
        if (self.cells_center is not None) and self.show_center:
            self._plot_cells_center(ax)
        if self.code2gene is not None:
            shapes, labels = self._plot_genes(ax)
            if legend_path:
                n = len(shapes)
                fig_legend, ax_legend = plt.subplots(figsize=(2, 0.23*n))
                ax_legend.legend(shapes, labels, loc=9)
                ax_legend.axis('off')
                fig_legend.tight_layout()
                fig_legend.savefig(legend_path)
            else:
                ax.legend(framealpha=0.5)
        if self.cell_assign is not None:
            self._plot_assign(ax)
        plt.ylim(0, self.img.shape[0])
        plt.xlim(0, self.img.shape[1])
        fig.tight_layout()
        if figpath:
            fig.savefig(figpath)
        else:
            plt.show()
        return self

    def _plot_genes(self, ax):
        marker_gen = marker_styles()
        s = 5 * (self.figsize[0] * self.figsize[1]) // 100
        shapes = []; labels = []
        for ix, gene in enumerate(self.code2gene.values()):
            pts = self.coordinates[ix][:, :2]
            if pts.shape[0] == 0:
                continue
            c, m = next(marker_gen)
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
