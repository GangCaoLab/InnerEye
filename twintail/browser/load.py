from os import path as osp
import logging
log = logging.getLogger(__file__)

from kivy.uix.popup import Popup

from .widgets.loader import LoadDialog
from .widgets.view_tab import ViewTab
from twintail.utils.io.h5 import list_datasets


class OpenedH5(object):

    def __init__(self):
        self.fpath = None

    def __get__(self, instance, owner):
        return self.fpath

    def __set__(self, wgt, fpath):
        """
        Behaviors when open a new .h5 file.
        """
        self.fpath = fpath
        import h5py
        with h5py.File(fpath, 'r') as f:
            dsets = list_datasets(fpath)
            first_dst = f[dsets[0]]
            imgs = [[ first_dst[:, :, 0, 0]  ]]
            wgt.draw2d(imgs)
            wgt.ids['z_slider'].slider.max = first_dst.shape[2] - 1
            wgt.ids['view_tab'].update_cycles(list(range(len(dsets))))
            wgt.ids['view_tab'].update_channels(list(range(first_dst.shape[3])))


class LoadMixin():

    opened = OpenedH5()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def show_load(self):
        dialog = LoadDialog(load=self.load, cancel=self.dismiss_popup, default_path=self.path)
        self._popup = Popup(title="Load file", content=dialog, size_hint=(0.8, 0.8))
        self._popup.open()

    def load(self, path, filenames):
        fpath = filenames[0]
        if osp.splitext(fpath)[1] not in {'.h5', '.hdf5'}:
            log.warning(f"Unsupported format: {fpath}")
            return
        self.opened = fpath
        if hasattr(self, '_popup'):
            self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()
