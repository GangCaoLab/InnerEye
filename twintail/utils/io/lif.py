import typing as t
import numpy as np


def read_series_uri(uri:str) -> np.ndarray:
    return read_series(*parse_uri(uri))


def parse_uri(uri:str) -> t.Tuple[str, str]:
    path, series = uri.split("::")
    return path, series


def read_series(path:str, series:t.Union[str, int]) -> np.ndarray:
    """Read series from a .lif file.

    :param path: Path to the .lif file.
    :param series: series to read.
    :return: Four dimensional array with shape (height, width, depth, channels)
    """
    import bioformats
    import javabridge

    javabridge.start_vm(class_path=bioformats.JARS)
    JVM_STARTED = True

    reader = bioformats.ImageReader(path)
    _imgs = []
    z = 0
    while 1:
        try:
            i = reader.read(z=z, series=series)
            _imgs.append(i)
            z += 1
        except javabridge.JavaException:
            break
    arr = np.c_[_imgs]
    arr = arr.swapaxes(0, 2).swapaxes(0, 1)

    javabridge.kill_vm()
    return arr

