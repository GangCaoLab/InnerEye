import typing as t
from collections import OrderedDict as od
from functools import lru_cache
import numpy as np
from sklearn.neighbors import KDTree
import networkx as nx


class DistGraphDecode(object):

    def __init__(self, spots: t.List[t.List[np.ndarray]]):
        self.spots = spots
        self.kdtrees = od()
        self.make_trees()
        self.dist_graph = None

    def make_trees(self):
        for ixcy, chs in enumerate(self.spots):
            if ixcy == 0:  # skip first cycle
                continue
            for ixch, pts in enumerate(chs):
                self.kdtrees.setdefault(ixcy, od())
                self.kdtrees[ixcy][ixch] = KDTree(pts)

    def query(self, X, radius, ixcy, ixch):
        tree = self.kdtrees[ixcy][ixch]
        ind, dist = tree.query_radius(X, radius, return_distance=True)
        return ind, dist

    def decode(self,
               chidxs: t.List[int],
               radius: float) -> t.Tuple[np.ndarray, np.ndarray]:
        assert len(chidxs) > 0
        dg = DistGraph(self)
        for ixch in chidxs:
            dg.add_cycle(ixch, radius)
        self.dist_graph = dg
        chains, dists = dg.get_chains()
        inds = np.array([cha[0][1] for cha in chains], dtype=np.int)
        pts = self.spots[0][chidxs[0]][inds]
        return pts, np.array(dists, dtype=np.float)


class DistGraph(object):

    def __init__(self, dc: DistGraphDecode):
        self.dc = dc
        self.ixcy = 0
        self.x = None
        self.old_inds = None
        self.G = nx.Graph()

    def init_nodes(self, ind: np.ndarray):
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

    def add_cycle(self, ixch, radius):
        if self.ixcy == 0:
            self.x = self.dc.spots[self.ixcy][ixch]
            self.old_inds = np.arange(self.x.shape[0])
            self.init_nodes(self.old_inds)
        else:
            inds, dists = self.dc.query(self.x, radius, self.ixcy, ixch)
            for ix in range(inds.shape[0]):
                ind = inds[ix]
                dist = dists[ix]
                for ix_node in self.old_inds:
                    self.add_edges(ind, base_ix=ix_node, dists=dist)
            uq_inds = np.unique(np.hstack(inds))
            self.x = self.dc.spots[self.ixcy][ixch][uq_inds, :]
            self.old_inds = uq_inds
        self.ixcy += 1

    def get_chains(self) -> t.Tuple[t.List[t.List[t.Tuple[int, int]]],
                                    t.List[float]]:
        ccs: t.List[set] = list(nx.connected_components(self.G))
        chains = []
        dists = []
        for cc in ccs:
            heads = []
            tails = []
            for ic, n in cc:
                if ic == self.ixcy - 1:
                    tails.append((ic, n))
                elif ic == 0:
                    heads.append((ic, n))
            if (len(tails) == 0) or (len(heads) == 0):
                continue
            paths = [(nx.dijkstra_path(self.G, h, t), nx.dijkstra_path_length(self.G, h, t))
                     for h in heads for t in tails]
            chain, d_sum = min(paths, key=lambda t: t[1])
            chains.append(chain)
            dists.append(d_sum)
        return chains, dists
