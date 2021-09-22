import os
import signal
import sys
from typing import Optional

import darkdetect
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt5.QtGui import QColor, QColorConstants
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QTabWidget
from pyqtgraph import PlotWidget

from lib.logger import Logger, FinishedLogData


class LogPlotter(QObject):
    data_update_slot = pyqtSignal(object)

    def __init__(self, logger: Logger):
        super().__init__()

        self.logger = logger

        # noinspection PyUnresolvedReferences
        self.data_update_slot.connect(self.on_update_data)

        self.window: Optional[QMainWindow] = None
        self.plot_widgets: Optional[dict[str, PlotWidget]]
        self.gen_plots = None
        self.batch_plots = None

    def update(self):
        data = self.logger.get_finished_data()
        # noinspection PyUnresolvedReferences
        self.data_update_slot.emit(data)

    def on_update_data(self, data: FinishedLogData):
        if self.window is None:
            self.create_window()
            self.create_plots(data)

            assert self.window is not None
            self.window.show()

        self.update_plot_data(data)

    def on_auto_range_pressed(self):
        if self.plot_widgets is not None:
            for plot_widget in self.plot_widgets.values():
                plot_widget.autoPixelRange = True
                plot_widget.enableAutoRange(x=True, y=True)

    def create_window(self):
        set_pg_defaults()

        self.window = QMainWindow()
        self.window.setWindowTitle("kZero training progress")
        self.window.setWindowFlag(Qt.WindowCloseButtonHint, False)

        main_widget = QWidget()
        self.window.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        autoRangeButton = QPushButton("Reset view")
        control_layout.addWidget(autoRangeButton)
        autoRangeButton.pressed.connect(self.on_auto_range_pressed)

        control_layout.addStretch()

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

    def create_plots(self, data: FinishedLogData):
        all_types = [k for k, _ in data.gen_keys] + [k for k, _ in data.batch_keys]
        all_types_unique = list(dict.fromkeys(all_types))

        self.plot_widgets = {}

        self.gen_plots = {}
        self.gen_average_plots = {}
        self.batch_plots = {}

        for ty in all_types_unique:
            widget = PlotWidget()
            widget.addLegend()
            self.plot_widgets[ty] = widget
            self.tab_widget.addTab(widget, ty)

            num_colors = all_types.count(ty)
            colors_main = generate_distinct_colors(1, 1, num_colors)
            colors_extra = generate_distinct_colors(0.5, 0.8, num_colors)

            def plot_all_matching(target, keys, prefix, colors):
                nonlocal next_color
                for (curr_ty, k) in keys:
                    if ty != curr_ty:
                        continue
                    pen = pg.mkPen(colors[next_color])
                    target[(ty, k)] = self.plot_widgets[ty].plot(name=f"{prefix} {ty} {k}", pen=pen)
                    next_color += 1

            next_color = 0
            plot_all_matching(self.batch_plots, data.batch_keys, "Batch", colors_extra)
            next_color = 0
            plot_all_matching(self.gen_average_plots, data.batch_keys, "Mean", colors_main)

            # don't reset color here, use the leftover colors
            plot_all_matching(self.gen_plots, data.gen_keys, "Gen", colors_main)

    def update_plot_data(self, data: FinishedLogData):
        gen_axis = 0.5 + np.arange(len(data.gen_data))

        for i, k in enumerate(data.gen_keys):
            self.gen_plots[k].setData(gen_axis, data.gen_data[:, i])
        for i, k in enumerate(data.batch_keys):
            self.gen_average_plots[k].setData(gen_axis, data.gen_average_data[:, i])

        for i, k in enumerate(data.batch_keys):
            self.batch_plots[k].setData(data.batch_axis, data.batch_data[:, i])


class PlotterThread(QThread):
    def run(self):
        self.app = QApplication([])
        self.app.exec()


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


def generate_distinct_colors(s: float, v: float, n: int):
    return [QColor.fromHsvF(h, s, v) for h in np.linspace(0, 1, num=n, endpoint=False)]


def set_pg_defaults():
    if darkdetect.isDark():
        pg.setConfigOption('background', QColorConstants.DarkGray.darker().darker())
        pg.setConfigOption('foreground', QColorConstants.LightGray)
    else:
        pg.setConfigOption('background', QColorConstants.LightGray)
        pg.setConfigOption('foreground', QColorConstants.DarkGray)
    pg.setConfigOption('antialias', True)
