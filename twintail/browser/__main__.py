from os import path as osp

import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout

from kivy.config import Config
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '700')


from .load import LoadMixin
from .plot2d import Plot2DMixin, init_figure


class Root(FloatLayout,
           LoadMixin,
           Plot2DMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = None


class BrowserApp(App):

    def __init__(self, path="~"):
        super().__init__()
        self.path = path
        self.root_wgt = None

    def build(self):
        from kivy.lang import Builder
        self.root_wgt = root = Builder.load_file('main.kv')
        figure_wgt = root.ids['figure_wgt']
        figure_wgt.figure = init_figure()
        path = self.path
        if osp.isfile(path):
            dir_ = osp.dirname(path)
            root.load(dir_, [path])
            root.path = dir_
        else:
            root.path = path
        root.ids['navbar_wgt'].path = root.path
        return root


def run_browser(path:str="~"):
    """
    :param path: Path to browser's root directory or directly open a .h5 file.
    """
    path = osp.realpath(path)
    bsr = BrowserApp(path)
    bsr.run()


if __name__ == '__main__':
    import fire
    fire.Fire(run_browser)
