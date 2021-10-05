from typing import Dict, Tuple

import darkdetect
import numpy as np
import pyqtgraph as pg
import scipy.signal
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QColor, QColorConstants
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QTabWidget, \
    QSlider, QLabel
from pyqtgraph import PlotWidget

from lib.logger import Logger, LoggerData


class LogPlotter(QObject):
    update_logger_slot = pyqtSignal(object)
    update_control_slot = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.prev_data = None

        # noinspection PyUnresolvedReferences
        self.update_logger_slot.connect(self.on_update_logger)
        # noinspection PyUnresolvedReferences
        self.update_control_slot.connect(self.on_update_control)

        self.create_window()

        self.plot_widgets: Dict[str, PlotWidget] = {}
        self.plot_items: Dict[Tuple[str, str], PlotWidget] = {}

        self.on_update_control()

    def update(self, logger: Logger):
        # noinspection PyUnresolvedReferences
        self.update_logger_slot.emit(logger.finished_data())

    def create_window(self):
        set_pg_defaults()

        self.window = QMainWindow()
        self.window.setWindowTitle("kZero training progress")
        self.window.setWindowFlag(Qt.WindowCloseButtonHint, False)

        self.window.resize(800, 500)

        main_widget = QWidget()
        self.window.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        autoRangeButton = QPushButton("Reset view")
        control_layout.addWidget(autoRangeButton)
        autoRangeButton.pressed.connect(self.on_auto_range_pressed)

        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimum(0)
        self.smooth_slider.setMaximum(100)
        self.smooth_slider.setSingleStep(1)
        self.smooth_slider.setValue(5)
        self.smooth_slider.valueChanged.connect(self.update_control_slot)
        control_layout.addWidget(self.smooth_slider)

        self.batch_smooth_label = QLabel()
        control_layout.addWidget(self.batch_smooth_label)

        control_layout.addStretch(1)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.window.show()

    def on_auto_range_pressed(self):
        for plot_widget in self.plot_widgets.values():
            plot_widget.autoPixelRange = True
            plot_widget.enableAutoRange(x=False, y=False)
            plot_widget.enableAutoRange(x=True, y=True)

    def widget_for_group(self, g: str):
        if g in self.plot_widgets:
            return self.plot_widgets[g]

        widget = PlotWidget()
        widget.addLegend()
        self.plot_widgets[g] = widget
        self.tab_widget.addTab(widget, g)
        return widget

    def on_update_control(self):
        window_size = self.smooth_slider.value() * 2 + 1
        self.batch_smooth_label.setText(str(window_size))

        if self.prev_data is not None:
            self.on_update_logger(self.prev_data)

    def on_update_logger(self, data: LoggerData):
        self.prev_data = data

        if len(self.plot_items) != len(data.values):
            self.update_plot_items(data)

        window_size = self.smooth_slider.value() * 2 + 1
        self.update_data(data, window_size)

    def update_plot_items(self, data: LoggerData):
        self.plot_items = {}
        groups = list(dict.fromkeys(g for g, _ in data.values))
        keys_per_group = {g: dict.fromkeys(k for h, k in data.values if h == g) for g in groups}

        for g in groups:
            widget = self.widget_for_group(g)
            widget.clear()
            widget.addLegend()

            keys = keys_per_group[g]
            colors = generate_distinct_colors(1.0, 1.0, len(keys))

            for (k, color) in zip(keys, colors):
                pen = pg.mkPen(color)
                self.plot_items[(g, k)] = widget.plot(name=f"{g} {k}", pen=pen)

    def update_data(self, data: LoggerData, window_size: int):
        for (g, k), v in data.values.items():
            x, y = clean_data(data.axis, v, window_size)
            self.plot_items[(g, k)].setData(x, y)


def clean_data(axis, values, window_size: int):
    mask = np.isnan(values)

    axis = axis[~mask]
    values = values[~mask]

    if window_size == 1:
        clean_values = values
    else:
        clean_values = scipy.signal.savgol_filter(values, window_size, polyorder=2, mode="nearest")

    return axis, clean_values


def qt_app():
    app = QApplication([])
    return app


def generate_distinct_colors(s: float, v: float, n: int):
    return [QColor.fromHsvF(h, s, v) for h in np.linspace(0, 1, num=n, endpoint=False)]


def set_pg_defaults():
    if darkdetect.isDark():
        pg.setConfigOption('background', QColorConstants.DarkGray.darker().darker())
        pg.setConfigOption('foreground', QColorConstants.LightGray)
    else:
        pg.setConfigOption('background', QColorConstants.LightGray.lighter())
        pg.setConfigOption('foreground', QColorConstants.Black)
    pg.setConfigOption('antialias', True)
