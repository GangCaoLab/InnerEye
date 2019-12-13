from collections import Iterable
import numpy as np

def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes, np.ndarray)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x
