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


def write_spots(path: str,
                spots: t.List[t.List[np.ndarray]],
                dims: t.List[t.Tuple]):
    """Write coordinates of all spots."""
    with h5py.File(path, 'w') as f:
        for ixcy, chs in enumerate(spots):
            grp = f.create_group(f"cycle_{ixcy}")
            grp.attrs.update({'dimension': dims[ixcy]})
            for ixch, arr in enumerate(chs):
                grp.create_dataset(f"channel_{ixch}", data=arr)


def read_spots(path: str) -> t.Tuple[t.List[t.List[np.ndarray]],
                                     t.List[t.Tuple]]:
    """Load coordinates of all spots."""
    spots = []
    dimensions = []
    with h5py.File(path, 'r') as f:
        cycle_names = sorted(filter(lambda a: a.startswith('cycle_'), f))
        for cycle in cycle_names:
            spots.append([])
            grp = f[cycle]
            channel_names = sorted(filter(lambda a: a.startswith('channel_'), grp))
            for channel in channel_names:
                arr = grp[channel][()]
                spots[-1].append(arr)
            dim = tuple(grp.attrs['dimension'])
            dimensions.append(dim)
    return spots, dimensions


def write_meta(uri: str, meta_info: dict):
    """Write meta information to hdf5 file."""
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'a') as f:
        f[inner_path].attrs.update(meta_info)


def read_meta(uri: str) -> dict:
    """Read meta information from hdf5 file."""
    path, inner_path = parse_uri(uri)
    with h5py.File(path, 'r') as f:
        meta = dict(f[inner_path].attrs)
    return meta


def write_decode(path: str,
                 genes: t.List[str],
                 points_per_gene: t.List[np.ndarray],
                 dists_per_gene: t.List[np.ndarray],
                 barcodes_per_gene: t.Optional[t.List[str]],
                 chidxs_per_gene: t.Optional[t.List[str]]):
    """Write decode result to hdf5 file."""
    with h5py.File(path, 'w') as f:
        g_p = f.create_group("points")
        g_d = f.create_group("distances")
        for ix, gene in enumerate(genes):
            g_p.create_dataset(gene, data=points_per_gene[ix])
            g_d.create_dataset(gene, data=dists_per_gene[ix])
        if barcodes_per_gene:
            f.attrs['barcodes'] = barcodes_per_gene
        if chidxs_per_gene:
            f.attrs['channel_indexes'] = chidxs_per_gene


def read_decode(path: str) -> t.Tuple[
    t.List[str],
    t.List[np.ndarray],
    t.List[np.ndarray],
    t.Optional[t.List[str]],
    t.Optional[t.List[str]],
    ]:
    """Read decode result to hdf5 file."""
    genes = []
    points_per_gene = []
    dists_per_gene = []
    barcodes_per_gene = []
    chidxs_per_gene = []
    with h5py.File(path) as f:
        g_p = f['points']
        g_d = f['distances']
        for gene, points in g_p.items():
            genes.append(gene)
            points_per_gene.append(points[()])
            dists_per_gene.append(g_d[gene][()])
        if 'barcodes' in f:
            barcodes_per_gene = f.attrs['barcodes']
        if 'channel_indexes' in f:
            chidxs_per_gene = f.attrs['channel_indexes']
    return genes, points_per_gene, dists_per_gene, barcodes_per_gene, chidxs_per_gene

