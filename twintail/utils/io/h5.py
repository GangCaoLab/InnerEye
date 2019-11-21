import typing as t
from os import path as ospath
import h5py
import numpy as np


def parse_uri(uri:str) -> t.Tuple[str, str]:
    items = uri.split("::")
    path = items[0]
    inner_path = "" if len(items) < 2 else items[1]
    return path, inner_path


def write_array(uri:str, arr:np.ndarray):
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'a') as f:
        f.create_dataset(inner_path, data=arr)


def read_array(uri:str) -> np.ndarray:
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'r') as f:
        return f[inner_path].value


def list_datasets(uri:str) -> t.List[str]:
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'r') as f:
        inner_path = inner_path or '/'
        lis = list(f[inner_path])
    return lis

