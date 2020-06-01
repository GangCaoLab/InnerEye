from itertools import product

from .base import ChainTool, ImgIO, GenesIO
from ..lib.img.misc import get_img_2d

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


class Plot2d(ChainTool, ImgIO, GenesIO):

    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize
        self.img = None
        self.genes = None
        self.coordinates = None

    def background(self, cycle=0, channel='mean', z='mean'):
        assert type(cycle) is int
        assert (type(channel) is int) or (channel == 'mean')
        assert (type(z) is int) or (z == 'mean')
        im4d = self.cycles[cycle]
        self.img = get_img_2d(im4d, channel, z)
        return self

    def plot(self, figpath=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        if self.img is not None:
            ax.imshow(self.img, cmap='gray')
        if self.genes is not None:
            self._plot_genes(ax)
        if figpath:
            fig.savefig(figpath)
        else:
            plt.show()
        return self

    def _plot_genes(self, ax):
        marker_gen = marker_styles()
        s = 5 * (self.figsize[0] * self.figsize[1]) // 100
        for ix, gene in enumerate(self.genes):
            pts = self.coordinates[ix][:, :2]
            if pts.shape[0] == 0:
                continue
            c, m = next(marker_gen)
            ax.scatter(pts[:, 1], pts[:, 0],
                       c=[c for _ in range(pts.shape[0])],
                       marker=m,
                       s=s,
                       label=gene)
        ax.legend(framealpha=0.5)
        plt.ylim(0, self.img.shape[0])
        plt.xlim(0, self.img.shape[1])

