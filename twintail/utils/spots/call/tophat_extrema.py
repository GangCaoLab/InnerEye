import typing as t
import numpy as np
from skimage.morphology import extrema, white_tophat


def call_spots(roi: np.ndarray,
               h: float = 0.1,
               q: t.Optional[float] = None):
    t = white_tophat(roi)
    if q:
        h = np.quantile(t, q)
    spots = np.c_[np.where(extrema.h_maxima(t, h))]
    return spots

