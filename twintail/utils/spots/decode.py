import typing as t
from collections import OrderedDict as od
from functools import lru_cache
import numpy as np
from sklearn.neighbors.kd_tree import KDTree
import networkx as nx


class DistGraphDecode(object):

    def __init__(self, spots: t.List[t.List[np.ndarray]]):
        self.spots = spots
        self.kdtrees = od()
        self.make_trees()
        self.dist_graph = None

    def make_trees(self):
        for ixcy, chs in self.spots:
            if ixcy == 0:  # skip first cycle
                continue
            for ixch, pts in chs:
                self.kdtrees.setdefault(od())
                self.kdtrees[ixcy][ixch] = KDTree(pts)

    @lru_cache(20)
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
        d_sum = np.sum(dists, axis=1)
        ind = [cc[0] for cc in chains]
        pts = self.spots[0][chidxs[0]][ind]
        return pts, d_sum


class DistGraph(object):

    def __init__(self, dc: DistGraphDecode):
        self.dc = dc
        self.ixcy = 0
        self.x = None
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
        for i in ind:
            if dists is None:
                e = ((self.ixcy-1, base_ix), (self.ixcy, i))
            else:
                e = ((self.ixcy-1, base_ix), (self.ixcy, i), {'d': dists[i]})
            edges.append(e)
        self.G.add_edges_from(edges)

    def add_cycle(self, ixch, radius):
        if self.ixcy == 0:
            self.x = self.dc.spots[self.ixcy][ixch]
            self.init_nodes(np.arange(self.x.shape[0]))
        else:
            inds, dists = self.dc.query(self.x, radius, self.ixcy, ixch)
            for ix in range(inds.shape[0]):
                ind = inds[ix]
                dist = dists[ix]
                for ix_node in ind:
                    self.add_edges(ind, base_ix=ix_node, dists=dist)
        self.ixcy += 1

    def get_chains(self):
        pass
