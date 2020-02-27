import typing as t
import numpy as np

from twintail.lib.log import print_arguments
from twintail.lib.io.h5 import read_cycles, write_cycles

from logging import getLogger
log = getLogger(__file__)


def registration(path, outpath, ref_cycle=-1, channel='mean', z='ignore', method='orb-match', **kwargs):
    """Image registration, align images to one reference cycle.

    :param path: Path to input hdf5 file.
    :param outpath: Path to output hdf5 file.
    :param ref_cycle: Index of reference cycle.
    :param channel: Channel used for registration.
    :param z: How to deal with z axis, default: ignore.
    :param method: Method used for registration.
    :param \**kwargs: Extra arguments for registration method.
    """
    print_arguments(log.info)
    assert z in {'ignore'}, f"Unknow z: {z}"
    assert method in {'orb-match'}, f"Unknown method: {method}"

    log.info("Read cycles.")
    cycles_arrs = read_cycles(path)
    arrs_for_align = [extract_channel(arr, channel) for arr in cycles_arrs]
    if z == 'ignore':
        arrs_for_align = [arr.mean(2) for arr in arrs_for_align]

    log.info("Compute transforms.")
    ref_arr = arrs_for_align[ref_cycle]
    if method == 'orb-match':
        match_func = orb_match
        transform_func = orb_transform
    else:
        raise NotImplementedError
    transforms = [
        None if arr is ref_arr else match_func(arr, ref_arr, **kwargs)
        for arr in arrs_for_align
    ]

    log.info("Apply transforms.")
    aligned = []
    ref_arr = cycles_arrs[ref_cycle]
    for idx, arr in enumerate(cycles_arrs):
        if arr is ref_arr:
            aligned.append(arr)
        else:
            t = transforms[idx]
            cs = []  # [[...zs], ...channels]
            for c_idx in range(arr.shape[3]):
                zs = []
                for z_idx in range(arr.shape[2]):
                    a = arr[:, :, z_idx, c_idx]
                    n = transform_func(a, a, t)
                    zs.append(n)
                cs.append(np.stack(zs, axis=2))
            aligned.append(np.stack(cs, axis=3))
    log.info(f"Output.")
    write_cycles(outpath, aligned)


def extract_channel(arr: np.ndarray, channel: t.Union[int, str]) -> np.ndarray:
    if isinstance(channel, int):
        return arr[channel]
    elif isinstance(channel, str) and channel.isdigit():
        return arr[int(channel)]
    elif channel == 'mean':
        return arr.mean(axis=3)
    else:
        raise NotImplementedError(f"Unsupported channel: {channel}")


def orb_match(im1, im2, max_features=500, good_match_percent=0.15, draw_match=None):
    import cv2
    # convert image to uint8
    im1 = ((im1/im1.max())*256) .astype(np.uint8)
    im2 = ((im2/im2.max())*256) .astype(np.uint8)
    # detect ORB features, compute descriptor
    orb = cv2.ORB_create(max_features)
    kpts1, desc1 = orb.detectAndCompute(im1, None)
    kpts2, desc2 = orb.detectAndCompute(im2, None)
    # match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    # only keep good matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    if draw_match:
        import matplotlib.pyplot as plt
        im_matches = cv2.drawMatches(im1, kpts1, im2, kpts2, matches, None)
        plt.figure(figsize=(20, 10))
        plt.imshow(im_matches)
        if isinstance(draw_match, str) and '.' in draw_match:
            plt.savefig(draw_match)
        else:
            plt.show()

    # extract location
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        pts1[i, :] = kpts1[match.queryIdx].pt
        pts2[i, :] = kpts2[match.trainIdx].pt
    # find homography
    h, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return h


def orb_transform(im, im_ref, h):
    import cv2
    height, width = im_ref.shape
    im_reg = cv2.warpPerspective(im, h, (width, height))
    return im_reg


if __name__ == "__main__":
    from twintail.lib.log import set_global_logging
    set_global_logging()
    import fire
    fire.Fire()
