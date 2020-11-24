import os
import typing as t
import logging
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from skimage.morphology import extrema, white_tophat


from ...img.transform import bright_range_transform
from ..cluster import merge_close_points
from ...log import TMP_DIR


log = logging.getLogger(__name__)


def get_tmp_dir() -> str:
    tmp_dir = f"{TMP_DIR}/call_spots"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    return tmp_dir


def find_optimize_br(
        im2d: np.ndarray,
        base_br_range=(30, 50),
        step=10,
        q=0.99,
        cc_size_range=(1, 9),
    ) -> t.Tuple[
        t.Tuple[int, int],
        np.ndarray,
        np.ndarray,
    ]:
    br = base_br_range
    best_num = -float('inf')
    best_br = None
    best_img = None
    best_centers = None
    while br[1] < 255:
        im2d_t = bright_range_transform(im2d, br)
        mask = im2d_t > np.quantile(im2d_t, q)
        regions = regionprops(label(mask))
        centers = np.array([
            cc.centroid for cc in regions if
            (cc_size_range[0] <= cc.area <= cc_size_range[1])
        ])
        if centers.shape[0] > best_num:
            best_num = centers.shape[0]
            best_br = br
            best_img = im2d_t
            best_centers = centers
        br = (br[0]+step, br[1]+step)

    return best_br, best_img, best_centers


def call_spots(roi: np.ndarray,
               base_br_range=(30, 50),
               spots_find_func=find_spots,
               spots_args=[],
               range_step=20,
               debug=False,
               ) -> np.ndarray:
    """
    Do bright_range_transform for multiple BR,
    then call spots for each BR,
    then merge them.
    """
    spots = np.array()

    if debug:
        # plot some figures for debug
        pass

    return spots
