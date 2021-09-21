import random

import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColorConstants
from PyQt5.QtWidgets import QApplication
from pyqtgraph import PlotItem


def main():
    app = QApplication([])
    item = PlotItem()

    widget = pg.plot(title="Title")
    widget.addLegend()

    a = widget.plot(name="a")
    b = widget.plot(name="b")

    a_data = [1.0]
    b_data = [1.0]

    def f():
        a_data.append(a_data[-1] * random.uniform(0.9, 1.1))
        b_data.append(b_data[-1] * random.uniform(0.9, 1.1))
        a.setData(a_data)
        b.setData(b_data)

    timer = QTimer()
    timer.timeout.connect(f)
    timer.start(1000/60)

    item.show()
    app.exec()


if __name__ == '__main__':
    main()
