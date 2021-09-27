import itertools
import os
import random
import shutil
from multiprocessing.pool import ThreadPool
from threading import Thread
from typing import Tuple

import numpy as np
from torch.optim import SGD

from lib.dataset import GameDataFile
from lib.games import Game
from lib.logger import Logger
from lib.loop import Buffer
from lib.plotter import LogPlotter, start_qt_app
from lib.save_onnx import save_onnx
from lib.train import TrainSettings, WdlTarget, WdlLoss, batch_loader
from lib.util import DEVICE, print_param_count


def map(args: Tuple[Game, str]):
    game, f = args
    GameDataFile(game, f).close()


def find_all_files(game: Game, folder: str, pre_map: bool):
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".bin.gz")]

    if pre_map:
        # TODO switch this back to multiprocessing once the "precess termination" issue with h5py is resolved
        pool = ThreadPool()
        args = [(game, p) for p in paths]
        pool.imap_unordered(map, args)

    return paths


def thread_main(logger: Logger, plotter: LogPlotter):
    data_folder = f"../../data/pgn-games/cclr/test/"
    network_folder = "../../data/supervised/initial/"

    shutil.rmtree(network_folder, ignore_errors=True)
    os.makedirs(network_folder, exist_ok=True)

    game = Game.find("chess")
    paths = find_all_files(game, data_folder, pre_map=False)

    batch_size = 1024

    settings = TrainSettings(
        game=game,
        wdl_target=WdlTarget.Final,
        wdl_loss=WdlLoss.MSE,
        policy_weight=100.0,
        batch_size=batch_size,
        batches=16,
        clip_norm=np.inf,
    )

    def block():
        return ResBlock(game, 32, 32, True, False, False, None, False)

    network = TowerModel(game, 32, 8, 32, True, True, True, block)
    network.to(DEVICE)

    print_param_count(network)

    # TODO weight decay?
    # TODO SDG vs Adam?
    optimizer = SGD(network.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    buffer_size = 1_000_000
    buffer = Buffer(game, int(buffer_size), 0.05)

    path_iter = itertools.cycle(paths)

    # pre-fill buffer
    for path in path_iter:
        buffer.append(None, path)
        if len(buffer.full_train_dataset()) >= buffer_size:
            break

    for gi in itertools.count():
        print("Starting gen {gi}")
        logger.start_gen()

        # append a random buffer each time as a coarse form of shuffling
        buffer.append(logger, random.choice(paths))

        test_loader = batch_loader(buffer.full_test_dataset(), batch_size)
        train_loader = batch_loader(buffer.full_test_dataset(), batch_size)

        settings.run_train(buffer.full_train_dataset(), optimizer, network, logger)

        network.eval()
        test_test_batch = next(iter(test_loader)).to(DEVICE)
        settings.evaluate_loss(network, "test-test", logger.log_gen, test_test_batch)
        test_train_batch = next(iter(train_loader)).to(DEVICE)
        settings.evaluate_loss(network, "test-train", logger.log_gen, test_train_batch)

        save_onnx(game, os.path.join(network_folder, f"network_{gi}.onnx"), network)

        logger.finish_gen()
        plotter.update()


def main():
    logger = Logger()
    plotter = LogPlotter(logger)

    app = start_qt_app()
    thread = Thread(target=thread_main, args=(logger, plotter))
    thread.start()
    app.exec()


if __name__ == '__main__':
    main()
