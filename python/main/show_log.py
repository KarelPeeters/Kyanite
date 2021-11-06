from lib.logger import Logger
from lib.plotter import qt_app, LogPlotter


def show_log(path: str):
    logger = Logger.load(path)

    app = qt_app()
    plotter = LogPlotter()
    plotter.update(logger)
    app.exec()


if __name__ == '__main__':
    show_log("../../data/supervised/lichess_09_2000_no_pov/log.npz")
