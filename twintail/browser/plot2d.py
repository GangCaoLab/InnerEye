from kivy.properties import ObjectProperty

from .widgets.mpl import MatplotFigure, MatplotNavToolbar
from .widgets.zslider import ZSlider

from collections import namedtuple
from dataclasses import dataclass, fields, astuple
import typing as t
from collections import Iterable
from functools import lru_cache

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def init_figure():
    plt.style.use('dark_background')
    mpl.rc('image', cmap='gray')
    fig, ax = plt.subplots()
    ax.text(
        -1, -2,
        'Please open a ".h5" file.',
        fontsize=20,
    )
    ax.set_xlim(-2, 10)
    ax.set_ylim(-8, 10)
    plt.arrow(3.6, 0, 3, 3,
        width=0,
        head_width=0.5,
        head_length=1,
        length_includes_head=True,
    )
    ax.text(
        7, 2,
        "File\n-> Open",
        fontsize=20
    )

    return fig


@dataclass()
class PlotInfo2D():
    z: int
    projection: bool
    cycles: t.List[int]
    channels: t.List[int]

    def frozen(self):
        _fields = [f.name for f in fields(self)]
        INFO = namedtuple("_PlotInfo2D", _fields)
        info = INFO(*[(tuple(e) if isinstance(e, list) else e) for e in astuple(self)])
        return info


class Plot2DMixin():

    plot_info = ObjectProperty(
        PlotInfo2D(
            0,
            False,
            [0],
            [0]
    ))

    def __init__(self, *args, **kwargs):
        self._nimgs = 0
        super().__init__(*args, **kwargs)

    @property
    def figure_wgt(self):
        return self.ids['figure_wgt']

    def _update(self, fig):
        self.figure_wgt.figure = fig

    @lru_cache(4)
    def _fetch_cycle(self, cycle_idx):
        with h5py.File(self.opened, 'r') as f:
            cycles = sorted(list(f))
            img4d = f[cycles[cycle_idx]].value
        return img4d

    @lru_cache(20)
    def _fetch_imgs(self, info):
        """
        Fetch 2D images.
        :return: [ [cycle1_channel1, cycle1_channel2, ...],
                   [cycle2_channel1, ...], ...]
        """
        if not self.opened:
            return []
        imgs = []

        z = info.z
        for cy_idx in info.cycles:
            imgs.append([])
            img4d = self._fetch_cycle(cy_idx)
            for ch_idx in info.channels:
                if info.projection:
                    img = img4d.mean(axis=2)[:, :, ch_idx]
                else:
                    img = img4d[:, :, z, ch_idx]
                imgs[-1].append(img)

        return imgs

    def on_plot_info(self, obj, val):
        imgs = self._fetch_imgs(self.plot_info.frozen())
        if list(flatten(imgs)):
            self.draw2d(imgs)

    def draw2d(self, imgs):
        n_cy = len(imgs)
        n_ch = min([len(l) for l in imgs])
        n = (n_cy, n_ch)
        if n != self._nimgs:
            fig, axes = plt.subplots(ncols=n_ch, nrows=n_cy)
            self._update(fig)
            self._nimgs = n
        else:
            axes = self.figure_wgt.figure.axes
        axes = axes.tolist() if isinstance(axes, np.ndarray) else axes
        axes = axes if isinstance(axes, Iterable) else [axes]
        axes = list(flatten(axes))
        for i, img in enumerate(flatten(imgs)):
            axes[i].imshow(img)
        self.figure_wgt.figure.canvas.draw()


def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes, np.ndarray)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x
