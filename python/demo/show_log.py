from lib.logger import Logger, FinishedLogData
from lib.plotter import start_qt_app, LogPlotter


def show_log(path: str):
    logger = Logger.from_finished_data(FinishedLogData.load(path))

    app = start_qt_app()
    plotter = LogPlotter(logger)
    plotter.update()
    app.exec()


if __name__ == '__main__':
    show_log("../../data/new_loop/test/log.npz")
