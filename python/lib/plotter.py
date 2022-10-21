import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Dict, Tuple, Callable, Optional

import darkdetect
import numpy as np
import pyqtgraph as pg
import scipy.signal
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt5.QtGui import QColor, QColorConstants
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QTabWidget, \
    QSlider, QLabel, QApplication
from pyqtgraph import PlotWidget

from lib.logger import Logger, LoggerData


class DummyLogPlotter:
    def update(self, _):
        pass

    def block_while_paused(self):
        return


class LogPlotterWindow(QObject):
    signal_smooth_window_size_changed = pyqtSignal(int)
    signal_pause_pressed = pyqtSignal()
    signal_reset_view_pressed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.window = QMainWindow()
        self.window.setWindowFlag(Qt.WindowCloseButtonHint, False)

        self.window.resize(800, 500)

        main_widget = QWidget()
        self.window.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        self.pauseButton = QPushButton("Pause")
        control_layout.addWidget(self.pauseButton)
        self.pauseButton.setEnabled(False)
        self.pauseButton.pressed.connect(self.signal_pause_pressed)

        autoRangeButton = QPushButton("Reset view")
        control_layout.addWidget(autoRangeButton)
        autoRangeButton.pressed.connect(self.signal_reset_view_pressed)

        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimum(0)
        self.smooth_slider.setMaximum(100)
        self.smooth_slider.setSingleStep(1)
        self.smooth_slider.setValue(5)
        self.plot_current_smoothing = 5
        self.smooth_slider.valueChanged.connect(self._on_smooth_slider_value_changed)
        control_layout.addWidget(self.smooth_slider)

        self.smooth_label = QLabel()
        control_layout.addWidget(self.smooth_label)

        control_layout.addStretch(1)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

    def show(self):
        self.window.show()

    def set_title(self, title: str):
        self.title = title
        QTimer.singleShot(0, self._actually_set_title)

    def _actually_set_title(self):
        if self.title is not None:
            self.window.setWindowTitle(self.title)

    def _on_smooth_slider_value_changed(self):
        value = self.smooth_slider.value() * 2 + 1
        self.smooth_label.setText(str(value))

        # noinspection PyUnresolvedReferences
        self.signal_smooth_window_size_changed.emit(value)


@dataclass
class PlotState:
    data: Optional[LoggerData]
    smooth_window_size: Optional[int]

    def __eq__(self, other):
        return self is other


class LogPlotter(QObject):
    signal_state_changed_to = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        set_pg_defaults()

        self.state_lock = Lock()
        self.state_drawn = PlotState(None, None)
        self.state_latest = PlotState(None, None)

        self.running = Event()
        self.running.set()

        self.window = LogPlotterWindow()
        # noinspection PyUnresolvedReferences
        self.window.signal_pause_pressed.connect(self._on_pause_pressed)
        # noinspection PyUnresolvedReferences
        self.window.signal_smooth_window_size_changed.connect(self._on_smooth_window_size_changed)
        # noinspection PyUnresolvedReferences
        self.window.signal_reset_view_pressed.connect(self._on_reset_view_pressed)

        # noinspection PyUnresolvedReferences
        self.signal_state_changed_to.connect(self._on_state_changed_to)

        self.plot_widgets: Dict[str, PlotWidget] = {}
        self.plot_items: Dict[Tuple[str, str], PlotWidget] = {}

        self.window._on_smooth_slider_value_changed()
        self.window.show()

    def set_title(self, title: str):
        self.window.set_title(title)

    def update(self, logger: Logger):
        data = logger.finished_data()
        with self.state_lock:
            state = PlotState(data, self.state_latest.smooth_window_size)
            self.state_latest = state
        # noinspection PyUnresolvedReferences
        self.signal_state_changed_to.emit(state)

    def set_can_pause(self, can_pause: bool):
        self.window.pauseButton.setEnabled(can_pause)

    def block_while_paused(self):
        self.set_can_pause(True)
        self.running.wait()

    def _on_pause_pressed(self):
        if self.running.is_set():
            self.running.clear()
            self.window.pauseButton.setText("Resume")
        else:
            self.running.set()
            self.window.pauseButton.setText("Pause")

    def _on_smooth_window_size_changed(self, smooth_window_size: int):
        with self.state_lock:
            state = PlotState(self.state_latest.data, smooth_window_size)
            self.state_latest = state

        # noinspection PyUnresolvedReferences
        self.signal_state_changed_to.emit(state)

    def _on_reset_view_pressed(self):
        for plot_widget in self.plot_widgets.values():
            plot_widget.autoPixelRange = True
            plot_widget.enableAutoRange(x=False, y=False)
            plot_widget.enableAutoRange(x=True, y=True)

    def _on_state_changed_to(self, event_state: PlotState):
        with self.state_lock:
            state_drawn = self.state_drawn
            state_latest = self.state_latest

            if self.state_latest != event_state:
                # we're not responsible for handling this event, discard it
                #   if we don't do this the QT event queue can fill up with old events
                return

            self.state_drawn = state_latest

        # ignore outdated events
        if state_drawn == state_latest:
            return

        self._render_new_state(state_latest)

    def _render_new_state(self, state: PlotState):
        if state.data is None:
            return

        # crate new tabs and plots if necessary
        if len(self.plot_items) != len(state.data.values):
            self._update_plot_items(state.data)

        # update the plot data
        self._update_plot_data(state)

    def _update_plot_items(self, data: LoggerData):
        self.plot_items = {}
        groups = list(dict.fromkeys(g for g, _ in data.values))
        keys_per_group = {g: dict.fromkeys(k for h, k in data.values if h == g) for g in groups}

        for g in groups:
            widget = self._widget_for_group(g)
            widget.clear()
            widget.addLegend()

            keys = keys_per_group[g]
            colors = generate_distinct_colors(1.0, 1.0, len(keys))

            for (k, color) in zip(keys, colors):
                pen = pg.mkPen(color)
                self.plot_items[(g, k)] = widget.plot(name=f"{g} {k}", pen=pen)

    def _widget_for_group(self, g: str):
        if g in self.plot_widgets:
            return self.plot_widgets[g]

        widget = PlotWidget()
        widget.addLegend()
        self.plot_widgets[g] = widget
        self.window.tab_widget.addTab(widget, g)
        return widget

    def _update_plot_data(self, state: PlotState):
        for (g, k), v in state.data.values.items():
            x, y = clean_data(state.data.axis, v, state.smooth_window_size)
            self.plot_items[(g, k)].setData(x, y)


def clean_data(axis, values, smooth_window_size: int):
    mask = np.isnan(values)

    axis = axis[~mask]
    values = values[~mask]

    if smooth_window_size == 1:
        clean_values = values
    else:
        clean_values = scipy.signal.savgol_filter(values, smooth_window_size, polyorder=1, mode="nearest")

    return axis, clean_values


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


def run_with_plotter(target: Callable[[LogPlotter], None]):
    """
    Run the given function with a newly constructed `LogPlotter`.
    This ensures the QApplication and GUI elements are created on a new thread, which then becomes the QT event loop.
    If an exception is thrown the QT event loop is also stopped allowing the program to fully exit.
    """

    plotter: Optional[LogPlotter] = None
    lock = Lock()
    lock.acquire()

    def gui_main():
        nonlocal plotter
        app = QApplication([])

        plotter = LogPlotter()
        lock.release()

        app.exec()

    gui_thread = Thread(target=gui_main)

    try:
        gui_thread.start()
        lock.acquire()

        target(plotter)
        print("Main thread finished")

        # if target finishes, we still need to keep this thread alive to detect KeyboardInterrupt
        while True:
            time.sleep(1000.0)

    except BaseException as e:
        QApplication.quit()
        raise e


def show_log(path: str):
    logger = Logger.load(path)

    def f(plotter: LogPlotter):
        plotter.update(logger)
        plotter.set_title(path)

    run_with_plotter(f)
