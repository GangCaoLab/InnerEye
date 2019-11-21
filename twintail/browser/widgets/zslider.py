from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty


from kivy.lang import Builder
Builder.load_file('widgets/zslider.kv')


class ZSlider(BoxLayout):
    max = ObjectProperty(10)
    projection = ObjectProperty(False)
    slider = ObjectProperty(None)
    zlabel = ObjectProperty(None)

    def on_checkbox_click(self, checkbox, value):
        print("check:", checkbox, value)
        if value:  # projection
            self.max = self.slider.max
            self.slider.max = 0
        else:
            self.slider.max = self.max

    def on_slide_val_change(self, slider, value):
        print("slide:", value)
        z = int(value)
        self.zlabel.text = f"z: {z}"

