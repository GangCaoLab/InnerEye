import typing as t
import h5py
import numpy as np


def parse_uri(uri: str) -> t.Tuple[str, str]:
    """Split URI to a file system path and a hdf5 path."""
    items = uri.split("::")
    path = items[0]
    inner_path = "" if len(items) < 2 else items[1]
    return path, inner_path


def write_array(uri: str, arr: np.ndarray):
    """Write numpy array."""
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'a') as f:
        f.create_dataset(inner_path, data=arr)


def read_array(uri: str) -> np.ndarray:
    """Load numpy array from hdf5 file."""
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'r') as f:
        return f[inner_path][()]


def list_datasets(uri: str) -> t.List[str]:
    """List all datasets in a hdf5 file or group."""
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'r') as f:
        inner_path = inner_path or '/'
        lis = list(f[inner_path])
    return lis


def write_cycles(path: str, cycles_arr: t.List[np.ndarray]):
    """Store image of each cycles."""
    with h5py.File(path, 'w') as f:
        for idx, arr in enumerate(cycles_arr):
            f.create_dataset(f"cycle_{idx}", data=arr)


def read_cycles(path: str) -> t.List[np.array]:
    """Load images of each cycles."""
    cycles = []
    with h5py.File(path, 'r') as f:
        arr_names = sorted(filter(lambda a: a.startswith('cycle_'), f))
        for n in arr_names:
            arr = f[n][()]
            cycles.append(arr)
    return cycles


def write_spots(path: str, spots: t.List[t.List[np.ndarray]]):
    """Write coordinates of all spots."""
    with h5py.File(path, 'w') as f:
        for ixcy, chs in enumerate(spots):
            grp = f.create_group(f"cycle_{ixcy}")
            for ixch, arr in enumerate(chs):
                grp.create_dataset(f"channel_{ixch}", data=arr)


def read_spots(path: str) -> t.List[t.List[np.ndarray]]:
    """Load coordinates of all spots."""
    spots = []
    with h5py.File(path, 'r') as f:
        cycle_names = sorted(filter(lambda a: a.startswith('cycle_'), f))
        for cycle in cycle_names:
            spots.append([])
            grp = f[cycle]
            channel_names = sorted(filter(lambda a: a.startswith('cycle_'), f))
            for channel in channel_names:
                arr = grp[channel][()]
                spots[-1].append(arr)
    return spots
