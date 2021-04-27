import random
from collections import Iterable, OrderedDict, defaultdict
import typing as t
from itertools import repeat
import importlib
from functools import lru_cache

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import napari

from ..lib.log import print_arguments
from ..lib.misc import local_arguments
from ..lib.img.misc import slide_over_ch, get_img_3d
from ..lib.spots.call.tophat_extrema import call_spots as tophat_extrema
from .base import MaskIO, Resetable
from .preprocessing import PreProcessing
from logging import getLogger
from ..lib.spots.call.blob import call_spots as call_blob

log = getLogger(__file__)


def func_for_slide(func: t.Callable, args: t.Tuple, channels: t.List, return_none=False) -> t.Callable:
    """Construct the function for slide over whole image."""
    def wrap(img: np.ndarray,
             idx: t.Union[int, t.Tuple[int, int]]) -> np.ndarray:
        # split args to different channels
        ix_ch = idx if not isinstance(idx, tuple) else idx[0]
        args_ = []
        for a in args:
            if isinstance(a, list):
                p = a[ix_ch]
            else:
                p = a
            args_.append(p)
        log.debug(f"Run function with args: {args_}")
        if ix_ch in channels:  # run only when channel specified.
            res = func(img, *args_)
        else:
            res = None if return_none else img
        return res
    return wrap


def random_color():
    color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color


class ViewMask3D(object):
    def __init__(self):
        self.gene2color = {"all": "white"}
        self.viewer = None

    @staticmethod
    def __roll_im_for_view(im, dim=3):
        if dim == 3:
            m = np.rollaxis(im, 2, 0)
        elif dim == 4:
            m = np.rollaxis(im, 3, 0)
            m = np.rollaxis(m, 3, 1)
        else:
            raise ValueError(f"Only support 3 and 4 dimension.")
        return m

    def view3d_signal(self, ixcy=[0], ixch=[0,1,2,3], merge_ch=False):
        print_arguments(log.info)
        if not isinstance(ixcy, list):
            ixcy = [ixcy]
        viewer = self.get_viewer()
        for_view = []
        channel_names = []
        for icy in ixcy:
            cy = self.cycles[icy]
            if merge_ch:
                imch = get_img_3d(cy, ixch)[:, :, :, np.newaxis]
                cy4view = self.__roll_im_for_view(imch, dim=4)
                for_view.append(cy4view)
                channel_names.append(f"cy:{icy} ch:{ixch}")
            else:
                imch = cy[:,:,:,ixch]
                cy4view = self.__roll_im_for_view(imch, dim=4)
                for_view.append(cy4view)
                channel_names.extend([f"cy:{icy} ch:{ich}" for ich in ixch])
        for_view = np.concatenate(for_view)
        viewer.add_image(for_view, channel_axis=0, name=channel_names)
        return self

    def init_viewer(self):
        viewer = napari.Viewer()
        self.viewer = viewer
        return viewer

    def get_viewer(self):
        if self.viewer is None:
            return self.init_viewer()
        else:
            return self.viewer

    def view3d_mask(self, ixcy=[0, 1], ixch=[0, 1], label_mask=False, show_spots=False):
        print_arguments(log.info)
        from skimage.measure import label
        if not isinstance(ixcy, list):
            ixcy = [ixcy]
        if not isinstance(ixch, list):
            ixch = [ixch]
        viewer = self.get_viewer()
        for icy in ixcy:
            im4d = self.cycles[icy]
            mask_cy = self.masks[icy]
            for ich in ixch:
                # add mask label layer
                mask_ch = mask_cy[:,:,:,ich]
                mask4view = self.__roll_im_for_view(mask_ch).astype(np.int)
                if label_mask:
                    mask4view = label(mask4view)
                label_layer = viewer.add_labels(mask4view, name=f"mask cy:{icy} ch:{ich}")
                # add spots label layer
                if show_spots and hasattr(self, 'spots'):
                    im_spts = np.zeros(mask_ch.shape)
                    s = self.spots[icy][ich]
                    im_spts[s[:,0], s[:,1], s[:,2]] = 1
                    im_spts4view = self.__roll_im_for_view(im_spts)
                    label_layer = viewer.add_labels(im_spts4view, name=f"spots cy:{icy} ch:{ich}")
        return self

    def view3d_spots(self, ixcy=[0], ixch=[0]):
        print_arguments(log.info)
        if not hasattr(self, 'spots'):
            return
        if not isinstance(ixcy, list):
            ixcy = [ixcy]
        if not isinstance(ixch, list):
            ixch = [ixch]
        viewer = self.get_viewer()
        for icy in ixcy:
            for ich in ixch:
                im_ch = self.cycles[icy][:,:,:,ich]
                im_spts = np.zeros(im_ch.shape)
                s = self.spots[icy][ich]
                im_spts[s[:,0], s[:,1], s[:,2]] = 1
                im_spts4view = self.__roll_im_for_view(im_spts)
                #color = {i: random_color() for i in range(1,10)}
                label_layer = viewer.add_labels(im_spts4view, name=f"spots cy:{icy} ch:{ich}")
        return self

    def view_mean_along_z(self):
        print_arguments(log.info)
        im = []
        for icy, im4d in enumerate(self.cycles):
            imcy = im4d.mean(axis=0).mean(axis=0).mean(axis=1)
            im.append(imcy)
        im = np.stack(im)
        napari.view_image(im, name="mean along z")
        return self

    def get_puncta_color(self, gene_name):
        if gene_name in self.gene2color:
            return self.gene2color[gene_name]
        else:
            color = random_color()
            self.gene2color[gene_name] = color
            return color

    def set_gene2color_from_tsv(self, path):
        with open(path) as f:
            for line in f:
                items = line.strip().split()
                self.gene2color[items[0]] = items[1]

    def view3d_punctas(
            self, text="{values}\n{code} {gene}",
            point_size=1, point_color=None,
            text_size=8, text_color="green",
            text_anchor="upper_left",
            split_by_gene=False,
            select=None,
            exclude=None,
            ):
        print_arguments(log.info)
        from skimage.measure import regionprops_table
        viewer = self.get_viewer()
        punctas_group = defaultdict(list)
        assert (select is None) or (exclude is None)

        if split_by_gene:
            for p in self.punctas.values():
                if (select and (p.gene not in select)) and \
                   (exclude and (p.gene in exclude)):
                    continue
                punctas_group[p.gene].append(p)
        else:
            if select or exclude:
                punctas_group = {
                    'all':
                    [
                        p for p in self.punctas.values()
                        if (select and (p.gene in select)) or (exclude and (p.gene not in exclude))
                    ]
                }
            else:
                punctas_group = {'all': list(self.punctas.values())}

        for g, punctas in punctas_group.items():
            pos, labels, vals, codes, genes = [[] for _ in range(5)]
            for p in punctas:
                pos.append(p.center)
                labels.append(p.id)
                vals.append([["%.2f"%v for v in l] for l in p.values])
                codes.append(p.codes)
                genes.append(p.gene)
            properties = {
                'label': np.array(labels),
                'values': np.array(vals),
                'code': np.array(codes),
                'gene': np.array(genes)
            }
            text_params = {
                'text': text,
                'size': text_size,
                'color': text_color,
                'anchor': 'upper_left',
            }
            pos = np.array(pos)
            pos = pos[:, [2,0,1]]
            if point_color is None:
                point_color = self.get_puncta_color(g)
            viewer.add_points(
                pos,
                text=text_params,
                properties=properties,
                size=point_size,
                opacity=0.7,
                face_color=point_color,
                name=g,
            )
        return self


class Volumn(PreProcessing, ViewMask3D, MaskIO):
    """Deal with volumetric stuff.
    Puncta is something like spots but keep original pixel/voxel"""

    def __init__(self,
                 n_workers: int = 1,
                 record_num: int = 2):
        print_arguments(log.info)
        self.n_workers = n_workers
        self.cycles = None
        self.masks = None
        self.extra_imgs = {}
        self.punctas = None
        self.code2gene = None
        Resetable.__init__(self, ["cycles", "masks"], limit=record_num)
        ViewMask3D.__init__(self)

    def add_merged_cycle(self, merge_channel=False):
        print_arguments(log.info)
        for ixcy in range(1, len(self.cycles)):
            assert self.cycles[ixcy].shape == self.cycles[0].shape
        merged = sum(self.cycles) / len(self.cycles)
        if merge_channel:
            merged = merged.mean(axis=3, keepdims=True)
        self.set_new(self.cycles + [merged], "cycles")
        return self

    def __expand_cycle_channel(self, cycles, channels):
        if channels is None:
            channels = list(range(self.cycles[0].shape[-1]))
        if cycles is None:
            cycles = list(range(len(self.cycles)))
        return cycles, channels

    def call_mask_blob(self, 
             p: float = 0.9,
             percentile_size: int = 15,
             q: float = 0.9,
             min_obj_size: int = 3,
             cycles=None,
             channels=None,
            ):
        print_arguments(log.info)
        masks = []
        cycles, channels = self.__expand_cycle_channel(cycles, channels)
        def call_blob_(*args, return_blob=True):
            return call_blob(*args, return_blob=return_blob)
        call_blob_ = func_for_slide(call_blob_, (p, percentile_size, q, min_obj_size), channels)
        for ixcy, img in enumerate(self.cycles):
            if ixcy in cycles:
                blob = slide_over_ch(img, call_blob_, self.n_workers, stack=False)
                blob = np.stack(blob, axis=-1)
            else:
                blob = img
            masks.append(blob)
        self.set_new(masks, "masks")
        return self

    def call_mask_from_spots(self):
        print_arguments(log.info)
        masks = []
        for ixcy, pts_cy in enumerate(self.spots):
            im4d = self.cycles[ixcy]
            mask_cy = np.full(im4d.shape, 0, dtype=np.int)
            if pts_cy is not None:
                for ixch, pts in enumerate(pts_cy):
                    if pts is not None:
                        mask_cy[pts[:,0],pts[:,1],pts[:,2],ixch] = 1
            masks.append(mask_cy)
        self.set_new(masks, "masks")
        return self

    def merge_masks(self, channels=None):
        print_arguments(log.info)
        masks = []
        for ixcy, mask in enumerate(self.masks):
            if channels is None:
                m_ch = np.logical_or.reduce(mask)
            else:
                m_ch = np.logical_or.reduce(mask[:,:,:,channels])
            mask = np.concatenate([mask, m_ch], axis=3)
            masks.append(mask)
        self.set_new(masks, "masks")
        return self

    def morphology_op(self,
                func_name="erosion",
                selm_shape="ball",
                selm_radius=1,
                cycles=None, channels=None,
                target='masks',
               ):
        mor = importlib.import_module("skimage.morphology")
        print_arguments(log.info)
        cycles, channels = self.__expand_cycle_channel(cycles, channels)
        selm_func = getattr(mor, selm_shape)
        selm = selm_func(selm_radius)
        op_func = getattr(mor, func_name)
        process = func_for_slide(op_func, (selm,), channels)
        results = []
        for ixcy, img in enumerate(getattr(self, target)):
            if ixcy in cycles:
                res = slide_over_ch(img, process, self.n_workers, stack=False)
                res = np.stack(res, axis=-1)
            else:
                res = img
            results.append(res)
        self.set_new(results, target)
        return self
    
    def mask_op(self, *args, **kwargs):
        kwargs.update({'target': 'masks'})
        self.morphology_op(*args, **kwargs)

    def img_op(self, *args, **kwargs):
        kwargs.update({'target': 'cycles'})
        self.morphology_op(*args, **kwargs)

    def call_spots(self, h=0.1, q=None, cycles=None, channels=None):
        """Call spots using tophat-filter + h_maxima method."""
        print_arguments(log.info)
        cycles, channels = self.__expand_cycle_channel(cycles, channels)
        spots = []
        process = func_for_slide(tophat_extrema, (h, q), channels, return_none=True)
        for ixcy, img in enumerate(self.cycles):
            if ixcy in cycles:
                res = slide_over_ch(img, process, self.n_workers, stack=False)
            else:
                res = None
            spots.append(res)
        self.spots = spots
        return self

    def segmentate_signals(self, min_obj_size=5, cycles=None, channels=None):
        """Segmentate signals by spots. Set segmentated label to masks"""
        from skimage.segmentation import watershed
        from skimage.measure import label
        from skimage.morphology import remove_small_objects
        print_arguments(log.info)
        cycles, channels = self.__expand_cycle_channel(cycles, channels)
        masks = []
        for ixcy in range(len(self.cycles)):
            img = self.cycles[ixcy]
            mask = self.masks[ixcy]
            spts = self.spots[ixcy]
            if ixcy in cycles:
                mask_chs = []
                for ixch in range(img.shape[3]):
                    mask_ch = mask[:,:,:,ixch]
                    if ixch in channels:
                        im_ch = img[:,:,:,ixch]
                        s = spts[ixch]
                        center_ch = np.zeros(im_ch.shape)
                        center_ch[s[:,0],s[:,1],s[:,2]] = 1
                        # remove centers outside mask
                        center_ch[mask_ch == 0] = 0
                        center_label = label(center_ch)
                        seg_label = watershed(-im_ch, center_label, mask=mask_ch)
                        seg_label = remove_small_objects(seg_label, min_obj_size)
                        mask_chs.append(seg_label)
                        self.extra_imgs['segmentated_label'] = seg_label
                    else:
                        mask_chs.append(mask_ch)
                res = np.stack(mask_chs, axis=-1)
            else:
                res = mask
            masks.append(res)
        self.set_new(masks, "masks")
        return self

    def exposure_adjust(self,
                        func_name="rescale_intensity",
                        args=("image", (0,1)),
                        cycles=None, channels=None,
                        ):
        """Perfrom exposure adjustment."""
        exp = importlib.import_module("skimage.exposure")
        print_arguments(log.info)
        cycles, channels = self.__expand_cycle_channel(cycles, channels)
        adj_func = getattr(exp, func_name)
        process = func_for_slide(adj_func, args, channels)
        imgs = []
        for ixcy, img in enumerate(self.cycles):
            if ixcy in cycles:
                res = slide_over_ch(img, process, self.n_workers, stack=False)
                res = np.stack(res, axis=-1)
            else:
                res = img
            imgs.append(res)
        self.set_new(imgs, "cycles")
        return self

    def parse_barcodes(self, path):
        print_arguments(log.info)
        code2gene = {}
        with open(path) as f:
            for line in f:
                items = line.strip().split()
                code2gene[items[0]] = items[1]
        self.code2gene = code2gene
        return self

    def decode(self, cycles=[0,1,2], channels=[0,1,2,3], channel_names="ACGT", min_size=1):
        """Run decode process on each segmentated blobs(punctas)."""
        print_arguments(log.info)
        label_img = self.extra_imgs['segmentated_label']
        punctas = OrderedDict()
        for lb_ix in range(1, int(label_img.max())):
            idx = np.where(label_img == lb_ix)
            p = Puncta(lb_ix, idx, channels, channel_names)
            p.extract_pixels([self.cycles[i] for i in cycles])
            if p.size >= min_size:
                punctas[lb_ix] = p
        self.punctas = punctas
        return self

    def extract_punctas_pixels(self, cycles=[0,1,2]):
        print_arguments(log.info)
        for p in self.punctas.values():
            p.extract_pixels([self.cycles[i] for i in cycles])
        return self

    def barcode_correction(self, dist_thresh=1, n_possible=1):
        """Correct barcode to clostest barcode."""
        print_arguments(log.info)
        from textdistance import hamming
        from random import choice
        def find_closest(s):
            min_d = float('inf')
            possible = []
            for code in self.code2gene.keys():
                d = hamming(s, code)
                if d <= dist_thresh:
                    min_d = min(d, min_d)
                possible.append((d, code))
            possible = [t for t in possible if t[0] == min_d]
            return possible
        for id_, p in self.punctas.items():
            if p.gene == "Unknow":
                codes = find_closest(p.codes)
                if 0 < len(codes) <= n_possible:
                    dist, code = codes[0]
                    p.ori_codes = code
                    p.codes = code
                    p.gene = self.code2gene[code]
                    p.dist = dist
                else:
                    p.dist = float('inf')
            else:
                p.dist = 0

        return self

    def filter_punctas(self,
            size_limit=(1, 100),
            thresh_chmax_intensity=None,
            thresh_cycles_intensity_percent=None,
            ):
        """Filter called punctas."""
        print_arguments(log.info)
        punctas = OrderedDict()
        for id_, p in self.punctas.items():
            if size_limit is not None:
                if not (size_limit[0] <= p.size <= size_limit[1]):
                    continue
            if thresh_chmax_intensity is not None:
                if not all([max(chs) >= thresh_chmax_intensity for chs in p.values]):
                    continue
            if thresh_cycles_intensity_percent is not None:
                cy_means = [sum(chs) for chs in p.values]
                if min(cy_means) < sum(cy_means)*thresh_cycles_intensity_percent:
                    continue
            punctas[id_] = p
        self.punctas = punctas
        return self

    def write_decode_res(self, path, val_thresh=0.05, percent_thresh=False, read_gene=True):
        """Write decode results to file."""
        print_arguments(log.info)
        with open(path, 'w') as f:
            f.write("id\tsize\tcenter\tcode\tgene\tvalues\n")
            for lb_id, p in self.punctas.items():
                if read_gene:
                    code = p.read_out(val_thresh, percent_thresh)
                    p.gene = self.code2gene.get(code, "Unknow")
                else:
                    code = p.codes
                items = [
                    p.id,
                    p.size,
                    p.center,
                    code,
                    p.gene,
                    p.values,
                ]
                outline = "\t".join([str(i) for i in items]) + "\n"
                f.write(outline)
        return self


class Puncta(object):
    def __init__(self, id_: int,
                 img_idx: t.Tuple[np.ndarray, np.ndarray, np.ndarray],
                 channels: t.List[int],
                 channel_names: str,
                 aggregate_method: str = "mean",
                 ch_num_per_cy: int = 2,
                 ):
        self.id = id_
        self.img_idx = img_idx
        self.channels = channels
        self.channel_names = channel_names
        self.aggregate_method = aggregate_method
        self.ch_num_per_cy = ch_num_per_cy
        self.pixels = None
        self.codes = None
        self.gene = None

    def extract_pixels(self, cycles: t.List[np.ndarray]):
        pixels = []
        for imcy in cycles:
            pixels.append([])
            for ixch in self.channels:
                imch = imcy[:,:,:,ixch]
                vals = imch[self.img_idx]
                pixels[-1].append(vals)
        self.pixels = pixels

    def read_out(self, val_thresh=0.1, percent_thresh=False):
        codes = []
        ch_num = self.ch_num_per_cy
        ch_names = self.channel_names
        for chs in self.values:
            args = np.argsort(chs)[::-1][:ch_num]
            sorted_chs = sorted(chs, reverse=True)
            d1 = sorted_chs[0] - sorted_chs[1]
            d2 = sorted_chs[1] - sorted_chs[2]
            if d1 > d2:
                code = ch_names[args[0]] * ch_num
            else:
                code = "".join(sorted([ch_names[i] for i in args]))

            #args = np.argsort(chs)[::-1][:ch_num]
            #sorted_chs = sorted(chs, reverse=True)[:ch_num]
            #if not percent_thresh:
            #    diff = (max(sorted_chs) - min(sorted_chs))
            #else:
            #    diff = max(sorted_chs) / min(sorted_chs)
            #if diff >= val_thresh:
            #    code = ch_names[args[0]]*ch_num
            #else:
            #    code = "".join([ch_names[i] for i in args])
            #code = "".join(sorted(code))

            codes.append(code)
        codes = "".join(codes)
        self.codes = codes
        return codes

    @property
    def size(self):
        return self.img_idx[0].shape[0]

    @property
    @lru_cache(maxsize=1)
    def center(self):
        return tuple([np.mean(i) for i in self.img_idx])

    @property
    @lru_cache(maxsize=1)
    def values(self):
        values = []
        agg_func = eval(f"np.{self.aggregate_method}")
        for chs in self.pixels:
            values.append([])
            for ch in chs:
                values[-1].append(agg_func(ch))
        return values

    @property
    @lru_cache(maxsize=1)
    def chastity(self):
        res = []
        vals = self.values
        for chs in vals:
            sorted_chs = sorted(chs, reverse=True)
            d13 = (sorted_chs[0] - sorted_chs[-1])
            c = d13 / (d13 + (sorted_chs[self.ch_num_per_cy] - sorted_chs[-1]))
            res.append(c)
        return res

