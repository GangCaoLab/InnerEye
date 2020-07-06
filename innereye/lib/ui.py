import json
import typing as t
from collections import defaultdict

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent, fig):
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, tool, figsize, n_cols_max, legend_path, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.tool = tool
        fig = tool._draw(figsize, n_cols_max, legend_path)
        self.fig = fig
        self.canvas = sc = MplCanvas(self, fig)
        sc.mpl_connect('button_press_event', self.onclick)
        toolbar = NavigationToolbar(sc, self)
        self.toolbar = toolbar
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)
        ann_helpers = QtWidgets.QHBoxLayout()
        l_ann = QtWidgets.QLabel("Annotation")
        l_ann.setStyleSheet('QLabel {font: 20pt Helvetica MS;}')
        ann_helpers.addWidget(l_ann)
        self.rbtn_pos = QtWidgets.QRadioButton("position")
        self.rbtn_neg = QtWidgets.QRadioButton("negative")
        for r in self.rbtn_pos, self.rbtn_neg:
            r.setStyleSheet('QRadioButton {font: 20pt Helvetica MS;}'
                            'QRadioButton::indicator { width: 20px; height: 20px;};')
        ann_helpers.addWidget(self.rbtn_pos)
        ann_helpers.addWidget(self.rbtn_neg)
        btn_ann = QtWidgets.QPushButton("Save")
        btn_ann.setStyleSheet('QPushButton {font: 20pt Helvetica MS;}')
        btn_ann.clicked.connect(self.saveAnnDialog)
        ann_helpers.addWidget(btn_ann)
        layout.addLayout(ann_helpers)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.show()

        _tp = t.Mapping[int, t.Mapping[str, t.Dict[t.Tuple[int, int], plt.Circle]]]
        self.idx2points: _tp = defaultdict(lambda: defaultdict(dict))
        self.rbtn_pos.click()

    @property
    def pos_or_neg(self):
        if self.rbtn_pos.isChecked():
            return 'pos'
        else:
            return 'neg'

    def onclick(self, event):
        ax = event.inaxes
        if ax is None:
            return
        ax_idx = self.fig.axes.index(ax)
        pos = (int(event.xdata), int(event.ydata))
        if self.toolbar.mode:
            return
        points = self.idx2points[ax_idx][self.pos_or_neg]
        points_r = self.idx2points[ax_idx]['neg' if self.pos_or_neg == 'pos' else 'pos']
        color = 'g' if self.pos_or_neg == 'pos' else 'r'
        shape = plt.Circle(pos, radius=0.5, fc=color)
        if pos not in points:
            points[pos] = shape
            ax.add_patch(shape)
            if pos in points_r:
                points_r.pop(pos).remove()
        else:
            points.pop(pos).remove()
        self.canvas.draw()

    def anns_to_json(self):
        res = []
        for i, conf in enumerate(self.tool.confs):
            grp = {}
            grp['img_path'] = conf.img_path
            grp['cycle'] = conf.cycle
            grp['channel'] = conf.channel
            grp['z'] = conf.z
            tp2points = {tp: list(pts) for tp, pts in self.idx2points[i].items()}
            grp['points'] = tp2points
            res.append(grp)
        return json.dumps(res)

    def saveAnnDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "QFileDialog.getSaveFileName()",
            "./ann.json", "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
            with open(fileName, 'w') as f:
                f.write(self.anns_to_json())

