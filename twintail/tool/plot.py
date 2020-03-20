from itertools import product

from .base import ChainTool
from twintail.lib.io.h5 import read_decode
from twintail.lib.img.misc import get_img_2d

import matplotlib.pyplot as plt
from matplotlib import cm
from random import random


def marker_styles(cmap="hsv"):
    markers = [".", "v", "^", "<", ">",
               "1", "2", "3", "4", "s",
               "p", "P", "*", "+", "x"]
    ix = 0
    cmap_ = cm.get_cmap(cmap)
    while True:
        yield cmap_(random()), markers[ix % len(markers)]
        ix += 1


class Plot2d(ChainTool):

    read_image = ChainTool.read

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

    def read_gene(self, path):
        genes, points_per_gene, _, _, _ = read_decode(path)
        self.genes = genes
        self.coordinates = points_per_gene
        return self

    def plot(self, figpath):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(self.img, cmap='gray')
        marker_gen = marker_styles()
        s = 10 * (self.figsize[0] * self.figsize[1]) // 100
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
        fig.savefig(figpath)
        return self
