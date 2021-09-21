import os
import signal
import sys
from typing import Optional

import darkdetect
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QColorConstants, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyqtgraph import PlotWidget

from log.logger import Logger, FinishedLogData


def start_qt_app():
    # make ctrl+C exit the program more quickly
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # make exceptions terminate the program immediately
    old_hook = sys.excepthook

    def hook(*args):
        old_hook(*args)
        os.kill(os.getpid(), signal.SIGINT)

    sys.excepthook = hook

    app = QApplication([])
    return app


class PlotterThread(QThread):
    def run(self):
        self.app = QApplication([])
        self.app.exec()


class LogPlotter(QObject):
    def __init__(self, logger: Logger):
        super().__init__()

        self.logger = logger

        # noinspection PyUnresolvedReferences
        self.update_slot.connect(self.on_update)

        self.window: Optional[QMainWindow] = None
        self.widget: Optional[PlotWidget] = None
        self.gen_plots = None
        self.batch_plots = None

    update_slot = pyqtSignal(object)

    def update(self):
        data = self.logger.get_finished_data()
        # noinspection PyUnresolvedReferences
        self.update_slot.emit(data)

    def on_update(self, data: FinishedLogData):
        if self.window is None:
            self.create_window(data)

        self.update_view(data)

    def create_window(self, data: FinishedLogData):
        self.window = QMainWindow()

        if darkdetect.isDark():
            pg.setConfigOption('background', QColorConstants.DarkGray.darker().darker())
            pg.setConfigOption('foreground', QColorConstants.LightGray)
        else:
            pg.setConfigOption('background', QColorConstants.LightGray)
            pg.setConfigOption('foreground', QColorConstants.DarkGray)
        pg.setConfigOption('antialias', True)

        self.widget: PlotWidget = pg.plot(title="Title")
        self.widget.addLegend()

        num_colors = len(data.gen_keys) + len(data.batch_keys)
        colors_dark = generate_distinct_colors(1, 0.5, num_colors)
        colors_light = generate_distinct_colors(1, 1, num_colors)

        self.gen_plots = {k: self.widget.plot(name=k, pen=pg.mkPen(colors_light[i])) for i, k in enumerate(data.gen_keys)}
        self.gen_average_plots = {k: self.widget.plot(name=k, pen=pg.mkPen(colors_light[len(data.gen_keys) + i])) for i, k in enumerate(data.batch_keys)}
        self.batch_plots = {k: self.widget.plot(name=k, pen=pg.mkPen(colors_dark[len(data.gen_keys) + i])) for i, k in enumerate(data.batch_keys)}

        self.window.setCentralWidget(self.widget)
        self.window.show()

    def update_view(self, data: FinishedLogData):
        gen_axis = 0.5 + np.arange(len(data.gen_data))

        for i, k in enumerate(data.gen_keys):
            self.gen_plots[k].setData(gen_axis, data.gen_data[:, i])
        for i, k in enumerate(data.batch_keys):
            self.gen_average_plots[k].setData(gen_axis, data.gen_average_data[:, i])

        for i, k in enumerate(data.batch_keys):
            self.batch_plots[k].setData(data.batch_axis, data.batch_data[:, i])


def generate_distinct_colors(s: float, v: float, n: int):
    return [QColor.fromHsvF(h, s, v) for h in np.linspace(0, 1, num=n, endpoint=False)]
