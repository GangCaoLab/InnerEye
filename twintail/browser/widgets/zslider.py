from copy import copy

from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.app import App

from kivy.lang import Builder
Builder.load_file('widgets/zslider.kv')


class ZSlider(BoxLayout):
    max = ObjectProperty(10)
    projection = ObjectProperty(False)
    slider = ObjectProperty(None)
    zlabel = ObjectProperty(None)

    def on_checkbox_click(self, checkbox, value):
        if value:  # change slider max
            self.max = self.slider.max
            self.slider.max = 0
        else:
            self.slider.max = self.max

        app = App.get_running_app()
        info = copy(app.root.plot_info)
        info.projection = value
        app.root.plot_info = info

    def on_slide_val_change(self, slider, value):
        z = int(value)
        self.zlabel.text = f"z: {z}"
        app = App.get_running_app()
        info = copy(app.root.plot_info)
        info.z = z
        app.root.plot_info = info

