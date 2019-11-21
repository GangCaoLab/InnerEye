from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty

from kivy.lang import Builder
Builder.load_file('widgets/saver.kv')


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)
    default_path = ObjectProperty(None)
