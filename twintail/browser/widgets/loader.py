from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty


from kivy.lang import Builder
Builder.load_file('widgets/loader.kv')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    default_path = ObjectProperty(None)

