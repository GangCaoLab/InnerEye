import matplotlib.pyplot as plt

from .widgets.mpl import MatplotFigure, MatplotNavToolbar
from .widgets.zslider import ZSlider

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


class Plot2DMixin():

    def __init__(self, *args, **kwargs):
        self._nimgs = 0
        super().__init__(*args, **kwargs)

    @property
    def figure_wgt(self):
        return self.ids['figure_wgt']

    def _update(self, fig):
        self.figure_wgt.figure = fig

    def draw2d(self, imgs):
        n = len(imgs)
        if n != self._nimgs:
            fig, axes = plt.subplots(ncols=n)
            axes = [axes] if n == 1 else axes
            self._update(fig)
            self._nimgs = n
        else:
            axes = self.figure_wgt.figure.axes

        for i, ax in enumerate(axes):
            ax.imshow(imgs[i])

