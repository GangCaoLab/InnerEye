import typing as t
from collections import OrderedDict as od
import numpy as np
from sklearn.neighbors import KDTree
import networkx as nx

import logging

log = logging.getLogger(__file__)


class DistGraphDecode(object):

    def __init__(self, spots: t.List[t.List[np.ndarray]]):
        self.spots = None
        self.ch_start_pos = None
        self.ind2channel: t.List[np.ndarray] = None
        self._merge_chs(spots)
        self._num_cycles = len(spots)
        self.kdtrees = od()
        self.make_trees()
        self.dist_graph = None

    def _merge_chs(self, spots_):  # merge spots of all channels together for decode
        spots = []
        start_poses: t.List[t.List[int]] = []
        channels = []
        for chs in spots_:
            pts = np.concatenate(chs)
            spots.append(pts)
            channel = np.zeros(pts.shape[0])
            pos = 0
            starts = []
            for ixch, pts_ in enumerate(chs):
                channel[pos:] = ixch
                starts.append(pos)
                pos += pts_.shape[0]
            start_poses.append(starts)
            channels.append(channel)
        self.spots = spots
        self.ch_start_pos = start_poses
        self.ind2channel = channels

    def make_trees(self):
        for ixcy, ch_pts in enumerate(self.spots):
            if ixcy == 0:  # skip first cycle
                continue
            self.kdtrees[ixcy] = KDTree(ch_pts)

    def query(self, X, radius, ixcy) -> t.Tuple[np.ndarray, np.ndarray]:
        log.debug(f"query array with shape: {X.shape} radius: {radius}")
        tree = self.kdtrees[ixcy]
        ind, dist = tree.query_radius(X, radius, return_distance=True)
        return ind, dist

    def decode(self,
               radius: float,
               res_cycle: int = 0) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param radius: radius threshold of per cycle's query.
        :param res_cycle: Use which cycle's position as result coordinate
        :return pos: coordinate of decoded.
        :return ch_idxes: channel index of each decode chain.
        :return dists: chain's length of each decode.
        """
        assert self._num_cycles > 0
        dg = DistGraph(self)
        log.debug("Begin to build graph:")
        for ixcy in range(self._num_cycles):
            log.debug(f"Add cycle: {ixcy}")
            dg.add_cycle(radius)
        self.dist_graph = dg
        log.debug( "Begin to decode from graph. "
                  f"graph size: {len(dg.G.nodes)} nodes {len(dg.G.edges)} edges")
        chains, dists = dg.get_chains()
        inds = np.array([
            [cha[ixcy][1] for ixcy in range(self._num_cycles)]
            for cha in chains
        ], dtype=np.int)
        pos = self.spots[res_cycle][inds[:, res_cycle]]
        ch_idxes = np.array([self.ind2channel[i][inds[:, i]] for i in range(inds.shape[1])])
        log.debug(ch_idxes)
        ch_idxes = ch_idxes.T
        log.debug(ch_idxes)
        log.debug(f"Decode complete, result shape: {pos.shape}")
        return pos, ch_idxes, np.array(dists, dtype=np.float)


class DistGraph(object):

    def __init__(self, dc: DistGraphDecode):
        self.dc = dc
        self.ixcy = 0
        self.x = None
        self.old_inds = None
        self.G = nx.Graph()

    def init_nodes(self, ind: np.ndarray):
        # node representation: (cycle_index, index)
        nodes = [(self.ixcy, i) for i in ind]
        self.G.add_nodes_from(nodes)

    def add_edges(self,
                  ind: np.ndarray,
                  base_ix: int,
                  dists: t.Optional[np.ndarray] = None,
                  ):
        edges = []
        for ix, i in enumerate(ind):
            if dists is None:
                e = ((self.ixcy-1, base_ix), (self.ixcy, i))
            else:
                e = ((self.ixcy-1, base_ix), (self.ixcy, i), {'weight': dists[ix]})
            edges.append(e)
        self.G.add_edges_from(edges)

    def add_cycle(self, radius):
        if self.ixcy == 0:
            self.x = self.dc.spots[self.ixcy]
            self.old_inds = np.arange(self.x.shape[0])
            self.init_nodes(self.old_inds)
        else:
            # get new cycle points indexes and distances(with old)
            inds, dists = self.dc.query(self.x, radius, self.ixcy)
            for ix in range(inds.shape[0]):
                ind = inds[ix]
                dist = dists[ix]
                ix_old = self.old_inds[ix]
                self.add_edges(ind, base_ix=ix_old, dists=dist)
            uq_inds = np.unique(np.hstack(inds))
            self.x = self.dc.spots[self.ixcy][uq_inds, :]
            self.old_inds = uq_inds
        self.ixcy += 1

    def get_chains(self) -> t.Tuple[t.List[t.List[t.Tuple[int, int]]],
                                    t.List[float]]:
        """One shortest path per CC(connected component)
        which contain three cycles."""
        ccs: t.List[set] = list(nx.connected_components(self.G))
        # filter out CC which contain empth cycle
        def contain_empty_cy(cc):
            cy_cnt = [0 for i in range(self.ixcy)]
            for cy, _ in cc:
                cy_cnt[cy] += 1
            return 0 in cy_cnt
        ccs = [cc for cc in ccs if (len(cc) >= self.ixcy) and (not contain_empty_cy(cc))]
        log.debug(f"Connected components number: {len(ccs)}")
        chains = []
        dists = []
        for cc in ccs:
            log.debug(f"CC size: {len(cc)}")
            heads = []
            tails = []
            for ic, n in cc:
                if ic == self.ixcy - 1:
                    tails.append((ic, n))
                if ic == 0:
                    heads.append((ic, n))
            paths = [(nx.dijkstra_path(self.G, h, t), nx.dijkstra_path_length(self.G, h, t))
                     for h in heads for t in tails]
            log.debug(f"Total paths: {len(paths)}")
            chain, d_sum = min(paths, key=lambda t: t[1])
            log.debug(f"Shortest path length: {d_sum}")
            chains.append(chain)
            dists.append(d_sum)
        return chains, dists
