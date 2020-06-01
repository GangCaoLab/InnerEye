from os import path as osp
import pathlib
import json
import multiprocessing

from ..lib.io.h5 import write_array, read_array


import logging
log = logging.getLogger(__file__)


def lif2hdf5(src_uri: str, dst_uri: str):
    """
    Extract image from .lif file save to hdf5 dataset.

    :param src_uri: URI to series saved in .lif file,
    e.g. "./Project.lif::0" means the first series in "./Project.lif".
    :param dst_uri: URI to the dataset in hdf5 file. e.g. "./sample1.hdf5::fov1"
    """
    from ..lib.io.lif import read_series_uri
    log.info(f"Read series from: {src_uri}")
    img = read_series_uri(src_uri)
    log.info(f"Save image to: {dst_uri}")
    write_array(dst_uri, img)


def copy_h5_dataset(src_uri: str, dst_uri: str):
    """
    Copy hdf5 dataset to another place.

    :param src_uri: URI to source dataset.
    :param dst_uri: URI to target dataset.
    """
    log.info(f"Read dataset from: {src_uri}")
    img = read_array(src_uri)
    log.info(f"Save image to: {dst_uri}")
    write_array(dst_uri, img)


def _extract(src_uri, dst_uri):
    ext = osp.splitext(src_uri.split('::')[0])[1]
    if ext in {'.h5', '.hdf5'}:
        copy_h5_dataset(src_uri, dst_uri)
    elif ext == '.lif':
        lif2hdf5(src_uri, dst_uri)
        from ..lib.io.lif import stop_jvm
        stop_jvm()
    else:
        raise IOError(f"{src_uri} is in unsupported input format.")


def extract_samples(json_path: str, dst_dir: str):
    """
    Extract sample images indecated by json config file to hdf5 format.

    :param json_path: Path to config json.
    :param dst_dir: Target directory to save extracted images.
    """
    with open(json_path) as f:
        samples = json.load(f)

    for samp_name, fovs in samples.items():
        log.info(f"Begin processing sample: {samp_name}")
        samp_dir = osp.join(dst_dir, samp_name)

        pathlib.Path(samp_dir).mkdir(parents=True, exist_ok=True)

        for fov_name, fov in fovs.items():
            log.info(f"\tFoV: {fov_name}")
            dst_f = osp.join(samp_dir, fov_name+'.h5')
            for idx, src_uri in enumerate(fov['cycles']):
                dst_uri = f"{dst_f}::cycle_{idx}"
                # start another process, prevent JVM canot start again
                p = multiprocessing.Process(target=_extract, args=(src_uri, dst_uri))
                p.start()
                p.join()


if __name__ == "__main__":
    from ..lib.log import set_global_logging
    set_global_logging()
    import fire
    fire.Fire()
