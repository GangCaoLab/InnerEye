from copy import copy

from kivy.uix.stacklayout import StackLayout
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.app import App

from functools import partial

from kivy.lang import Builder
Builder.load_file("widgets/view_tab.kv")


class ViewTab(StackLayout):

    cycles_grid = ObjectProperty(None)
    channels_grid = ObjectProperty(None)

    DEFAULT_COLORS = ['#00ff00', '#00ffff', '#ff0000', '#800080', '#0000ff']

    def _fetch_active_idx(self, grid):
        widgets = grid.children[::-1]
        ncols = grid.cols
        nrows = len(widgets) // ncols
        channels = []
        for i in range(nrows):
            cb = widgets[i*ncols + 1] 
            if cb.active:
                channels.append(i)
        return channels

    def update_cycles(self, cycles):
        grid = self.cycles_grid

        def on_active(cb, value):
            app = App.get_running_app()
            info = copy(app.root.plot_info)
            info.cycles = self._fetch_active_idx(grid)
            app.root.plot_info = info
            navbar = app.root.ids['navbar_wgt']
            if navbar.link_btn.state == 'down':
                navbar._navtoolbar.unlink_axes()
            navbar.key_up()

        for i, name in enumerate(cycles):
            grid.add_widget(Label(text=str(i)))
            cb = CheckBox(active=False if i > 0 else True)
            cb.bind(active=on_active)
            grid.add_widget(cb)
        grid.height = len(cycles) * grid.row_default_height

    def update_channels(self, channels):
        grid = self.channels_grid

        def on_active(cb, value):
            app = App.get_running_app()
            info = copy(app.root.plot_info)
            info.channels = self._fetch_active_idx(grid)
            app.root.plot_info = info
            navbar = app.root.ids['navbar_wgt']
            if navbar.link_btn.state == 'down':
                navbar._navtoolbar.unlink_axes()
            navbar.key_up()

        for i, name in enumerate(channels):
            grid.add_widget(Label(text=str(i)))
            cb = CheckBox(active=False if i > 0 else True)
            cb.bind(active=on_active)
            grid.add_widget(cb)
            colors = self.DEFAULT_COLORS
            ti = TextInput(
                text=colors[i%len(colors)],
                foreground_color=(1, 1, 1, 1),
                background_color=(.1,.1,.1, 1),
            )
            grid.add_widget(ti)
        grid.height = len(channels) * grid.row_default_height
