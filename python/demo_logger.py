import itertools
import random
import time
from threading import Thread

from log.logger import Logger


def main_thread(logger: Logger, plotter):
    loss_a = 1.0
    loss_b = 1.0
    loss_c = 1.0

    for _ in itertools.count():
        logger.start_gen()

        loss_c *= random.uniform(0.9, 1.1)
        logger.log_gen("loss", "c", loss_c)

        for bi in range(20):
            logger.start_batch()

            loss_a *= random.uniform(0.9, 1.1)
            loss_b *= random.uniform(0.9, 1.1)
            logger.log_batch("loss", "a", loss_a)
            logger.log_batch("large", "a", loss_a * 1000)
            logger.log_batch("loss", "b", loss_b)

            logger.finish_batch()

        logger.finish_gen()
        plotter.update_data()

        time.sleep(0.2)


def main():
    from log.plotter import LogPlotter
    from log.plotter import start_qt_app
    app = start_qt_app()

    logger = Logger()
    plotter = LogPlotter(logger)

    thread = Thread(target=main_thread, args=(logger, plotter))
    thread.start()

    app.exec()


if __name__ == '__main__':
    main()
