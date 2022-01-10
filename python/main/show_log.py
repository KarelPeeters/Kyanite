from lib.logger import Logger
from lib.plotter import run_with_plotter


def show_log(path: str):
    logger = Logger.load(path)

    run_with_plotter(lambda plotter: plotter.update(logger))


if __name__ == '__main__':
    show_log("../../data/supervised/lichess_09_2000_no_pov/log.npz")
