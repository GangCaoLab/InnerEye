import numpy as np

def scale_to_255(img):
    img_ = img * 255 / img.max()
    return img_

def bright_range_transform(img, bright_range):
    """
    Scale the bright range (m, M) in the image
    to (0, 255).

     255 ---           --- 255
          |        /    |
          | M---        |
          |   |  scale  |
          | m---        |
          |        \    |
       0 ---           --- 0
    """
    m = img.copy()
    rg = bright_range
    m = m - rg[0]
    m = m * 255 / (rg[1] - rg[0])
    m[m > 255] = 255
    m[m < 0] = 0
    return m
