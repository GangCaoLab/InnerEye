"""
https://github.com/jeysonmc/kivy_matplotlib/blob/master/kivy_matplotlib.py
"""
import math
import os.path as osp

from kivy.uix.widget import Widget
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import NavigationToolbar2
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, ListProperty
from kivy.base import EventLoop
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup

from .saver import SaveDialog


import logging
log = logging.getLogger(__file__)


from kivy.lang import Builder
Builder.load_file("widgets/mpl.kv")


class MatplotFigure(Widget):

    """Widget to show a matplotlib figure in kivy.
    The figure is rendered internally in an AGG backend then
    the rgb data is obtained and blitted into a kivy texture.
    """

    figure = ObjectProperty(None)
    _box_pos = ListProperty([0, 0])
    _box_size = ListProperty([0, 0])
    _img_texture = ObjectProperty(None)
    _bitmap = None
    _pressed = False
    figcanvas = ObjectProperty(None)
    # I Chose composition over MI because of name clashes

    def on_figure(self, obj, value):
        self.figcanvas = _FigureCanvas(self.figure, self)
        self.figcanvas._isDrawn = False
        l, b, w, h = self.figure.bbox.bounds
        w = int(math.ceil(w))
        h = int(math.ceil(h))
        self.width = w
        self.height = h

        # Texture
        self._img_texture = Texture.create(size=(w, h))

    def __init__(self, figure=None, *args, **kwargs):
        super(MatplotFigure, self).__init__(*args, **kwargs)
        self.figure = figure
        # Event binding
        EventLoop.window.bind(mouse_pos=self.on_mouse_move)
        self.bind(size=self._on_size)

    def _draw_bitmap(self):
        if self._bitmap is None:
            log.info("No bitmap!")
            return
        self._img_texture = Texture.create(size=(self.bt_w, self.bt_h))
        self._img_texture.blit_buffer(
            self._bitmap, colorfmt="rgb", bufferfmt='ubyte')
        self._img_texture.flip_vertical()

    def on_mouse_move(self, window, mouse_pos):
        """ Mouse move """
        if self._pressed:  # Do not process this event if there's a touch_move
            return
        x, y = mouse_pos
        if self.collide_point(x, y):
            real_x, real_y = x - self.pos[0], y - self.pos[1]
            self.figcanvas.motion_notify_event(x, real_y, guiEvent=None)

    def on_touch_down(self, event):
        x, y = event.x, event.y

        if self.collide_point(x, y):
            self._pressed = True
            real_x, real_y = x - self.pos[0], y - self.pos[1]
            self.figcanvas.button_press_event(x, real_y, 1, guiEvent=event)

    def on_touch_move(self, event):
        """ Mouse move while pressed """
        x, y = event.x, event.y
        if self.collide_point(x, y):
            real_x, real_y = x - self.pos[0], y - self.pos[1]
            self.figcanvas.motion_notify_event(x, real_y, guiEvent=event)

    def on_touch_up(self, event):
        x, y = event.x, event.y
        if self._box_size[0] > 1 or self._box_size[1] > 1:
            self.reset_box()
        if self.collide_point(x, y):
            pos_x, pos_y = self.pos
            real_x, real_y = x - pos_x, y - pos_y
            self.figcanvas.button_release_event(x, real_y, 1, guiEvent=event)
            self._pressed = False

    def new_timer(self, *args, **kwargs):
        pass  # TODO

    def _on_size(self, o, size):
        if self.figure is None:
            return
        # Creat a new, correctly sized bitmap
        self._width, self._height = size
        self._isDrawn = False

        if self._width <= 1 or self._height <= 1:
            return

        dpival = self.figure.dpi
        winch = self._width / dpival
        hinch = self._height / dpival
        self.figure.set_size_inches(winch, hinch)
        self.figcanvas.resize_event()
        self.figcanvas.draw()

    def reset_box(self):
        self._box_size = 0, 0
        self._box_pos = 0, 0

    def draw_box(self, event, x0, y0, x1, y1):
        pos_x, pos_y = self.pos
        # Kivy coords
        y0 = pos_y + y0
        y1 = pos_y + y1
        self._box_pos = x0, y0
        self._box_size = x1 - x0, y1 - y0


class _FigureCanvas(FigureCanvasAgg):

    """Internal AGG Canvas"""

    def __init__(self, figure, widget, *args, **kwargs):
        self.widget = widget
        super(_FigureCanvas, self).__init__(figure, *args, **kwargs)
        super().mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        print(event, event.inaxes)

    def draw(self):
        """
        Render the figure using agg.
        """
        super(_FigureCanvas, self).draw()
        agg = self.get_renderer()
        w, h = agg.width, agg.height
        self._isDrawn = True

        self.widget.bt_w = w
        self.widget.bt_h = h
        self.widget._bitmap = agg.tostring_rgb()
        self.widget._draw_bitmap()

    def blit(self, bbox=None):
        # TODO bbox
        agg = self.get_renderer()
        w, h = agg.width, agg.height
        self.widget._bitmap = agg.tostring_rgb()
        self.widget.bt_w = w
        self.widget.bt_h = h
        self.widget._draw_bitmap()

    def print_figure(self, filename, *args, **kwargs):
        #super(self.print_figure, self).print_figure(filename, *args, **kwargs)
        super().print_figure(filename, *args, **kwargs)
        if self._isDrawn:
            self.draw()


class MatplotNavToolbar(BoxLayout):
    """Figure Toolbar"""
    pan_btn = ObjectProperty(None)
    zoom_btn = ObjectProperty(None)
    home_btn = ObjectProperty(None)
    link_btn = ObjectProperty(None)
    info_lbl = ObjectProperty(None)
    save_btn = ObjectProperty(None)
    path = ObjectProperty(None)

    _navtoolbar = None  # Internal NavToolbar logic
    figure_widget = ObjectProperty(None)

    def __init__(self, figure_widget=None, *args, **kwargs):
        super(MatplotNavToolbar, self).__init__(*args, **kwargs)
        self.figure_widget = figure_widget

    def on_figure_widget(self, obj, value):
        self.figure_widget.bind(figcanvas=self._canvas_ready)

    def _canvas_ready(self, fig, canvas):
        if self._navtoolbar is None:
            self._navtoolbar = _NavigationToolbar(canvas, self)
            self._navtoolbar.figure_widget = fig
        else:
            self._navtoolbar.__init__(canvas, self)
            self._navtoolbar.figure_widget = fig

    def key_up(self):
        for btn in ['pan', 'zoom', 'link']:
            getattr(self, btn+'_btn').state = 'normal'


class _NavigationToolbar(NavigationToolbar2):
    figure_widget = None

    def __init__(self, canvas, widget):
        self._link = False
        self.widget = widget
        super(_NavigationToolbar, self).__init__(canvas)

    def _init_toolbar(self):
        self.widget.home_btn.bind(on_press=self.home)
        self.widget.pan_btn.bind(on_press=self.pan)
        self.widget.zoom_btn.bind(on_press=self.zoom)
        self.widget.link_btn.bind(on_press=self.press_link)
        self.widget.save_btn.bind(on_press=self.press_save)

    def press_link(self, *args):
        if self._link:
            self.unlink_axes()
        else:
            self.link_axes()

    def link_axes(self, *args):
        self._link = True
        axes = self.figure_widget.figure.axes
        if len(axes) > 1:
            axes[0].get_shared_x_axes().join(*axes)
            axes[0].get_shared_y_axes().join(*axes)

    def unlink_axes(self):
        self._link = False
        axes = self.figure_widget.figure.axes
        if len(axes) > 1:
            for ax in axes[1:]:
                axes[0].get_shared_x_axes().remove(ax)
                axes[0].get_shared_y_axes().remove(ax)

    def press_save(self, *args):
        dialog = SaveDialog(save=self.save, cancel=self.dismiss_popup,
                            default_path=self.widget.path)
        self._popup = Popup(title="Save figure", content=dialog,
                            size_hint=(0.8, 0.8))
        self._popup.open()

    def save(self, path, filename):
        print(path)
        print(filename)
        fname = osp.join(path, filename)
        try:
            self.save_figure(fname)
            self.dismiss_popup()
        except Exception as e:
            log.warning(f"Faile to save {fname}: " + str(e))
            self._popup.content.text_input.background_color = (0.7, 0.01, 0.01, 0.5)

    def save_figure(self, fname):
        self.canvas.figure.savefig(fname)

    def dismiss_popup(self):
        self._popup.dismiss()

    def dynamic_update(self):
        self.canvas.draw()

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.figure_widget.draw_box(event, x0, y0, x1, y1)

    def set_message(self, s):
        self.widget.info_lbl.text = s

