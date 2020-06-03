from .preprocessing import PreProcessing
from .call_spots import CallSpots
from .spots_op import SpotsOp
from .decode import Decode
from .cells_op import CellsOp
from .plot import Plot2d


class Chain(PreProcessing, CallSpots, SpotsOp, Decode, CellsOp, Plot2d):
    """Chain all steps together."""
    def __init__(self, z_mode='slide', n_workers=1):
        super().__init__()
        self.z_mode = z_mode
        self.n_workers = n_workers

    def set_param(self, attr, val):
        setattr(self, attr, val)

    def set_params(self, params: dict):
        for attr, val in params.items():
            self.set_param(attr, val)
