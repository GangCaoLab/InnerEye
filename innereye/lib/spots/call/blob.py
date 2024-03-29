import typing as t
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


def blob_to_spot(blobs: np.ndarray) -> np.ndarray:
    regions = regionprops(label(blobs))
    centroids = [r.centroid for r in regions]
    return np.array(centroids)


def call_spots(roi: np.ndarray,
               p: float = 0.9,
               percentile_size: int = 15,
               q: float = 0.9,
               min_obj_size: int = 3,
               return_blob: bool = False,
               min_nn: t.Optional[int] = None,
               ) -> np.ndarray:
    """
    :param roi: Input image.
    :param p: Local mask threshold. (0, 1)
    :param percentile_size: Local region size.
    :param q: Global mask threshold. (0, 1)
    :param min_obj_size: Min object area size.
    :param min_nn: Min nearest neighbors.
    """
    dim = len(roi.shape)
    if dim == 2:
        min_nn = min_nn or 3
        nn_se = np.ones((3, 3))
    else:
        min_nn = min_nn or 7
        nn_se = np.ones((3, 3, 3))
    # change to filters.rank
    per = roi - ndi.percentile_filter(roi, p*100, percentile_size)
    th = np.quantile(roi, q)
    f = (per > 0) & (roi > th)
    blob = remove_small_objects(f, min_size=min_obj_size)
    blob = blob & (ndi.convolve(blob.astype(np.int), nn_se) > min_nn)
    blob = remove_small_objects(blob, min_size=min_obj_size)
    if return_blob:
        return blob
    else:
        spots = blob_to_spot(blob)
        return spots

