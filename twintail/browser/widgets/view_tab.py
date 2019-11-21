from kivy.uix.stacklayout import StackLayout
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty

from functools import partial

from kivy.lang import Builder
Builder.load_file("widgets/view_tab.kv")


class ViewTab(StackLayout):

    cycles_grid = ObjectProperty(None)
    channels_grid = ObjectProperty(None)

    DEFAULT_COLORS = ['#00ff00', '#00ffff', '#ff0000', '#800080', '#0000ff']

    def update_cycles(self, cycles):
        grid = self.cycles_grid
        def on_active(cb, value):
            print(value)
            cb.value = value
        for i, name in enumerate(cycles):
            grid.add_widget(Label(text=str(i)))
            cb = CheckBox(active=False if i > 0 else True)
            cb.bind(active=on_active)
            grid.add_widget(cb)
        grid.height = len(cycles) * grid.row_default_height

    def update_channels(self, channels):
        grid = self.channels_grid
        def on_active(cb, value):
            print(value)
            cb.value = value
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
