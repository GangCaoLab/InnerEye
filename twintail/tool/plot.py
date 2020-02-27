from .base import ChainTool, SpotsTool


class Plot(ChainTool):

    read_image = ChainTool.read
    read_spots = SpotsTool.read

