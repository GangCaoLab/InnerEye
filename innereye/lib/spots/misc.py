import typing as t
import numpy as np
from skimage.measure import label, regionprops


def coordinates_to_mask(points: np.ndarray,
                        shape: t.Optional[t.Tuple] = None) -> np.ndarray:
    """Convert coordinates to mask array.

    :param points: Coordinates of all points.
    Will be converted to the dtype np.int.
    :param shape: Shape of mask array.
    :return: Mask array with dtype bool.
    """
    points = points.astype(np.int)
    dim_max = tuple([points[:, i].max()+1 for i in range(points.shape[1])])
    if shape is None:
        shape = dim_max
    else:
        assert len(shape) == points.shape[1]
        shape = tuple([shape[i] or dim_max[i] for i in range(points.shape[1])])
    arr = np.zeros(shape, dtype=np.bool)
    ix = tuple(points[:, d] for d in range(points.shape[1]))
    arr[ix] = True
    return arr


def cc_centroids(mask: np.ndarray) -> np.ndarray:
    """Get centroids of all connected components in mask.

    :param mask: Input mask.
    :return: Coordinates of all connected components center.
    """
    ccs = regionprops(label(mask))
    centroids = np.array([cc.centroid for cc in ccs])
    return centroids

