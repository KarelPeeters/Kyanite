import itertools
import os
import random
import shutil
from multiprocessing.pool import ThreadPool
from threading import Thread
from typing import Tuple

import numpy as np
import torch.nn.functional as nnf
from torch.optim import AdamW
from torch.utils.data import Subset

from lib.dataset import GameDataFile
from lib.dataview import GameDataView
from lib.games import Game
from lib.logger import Logger
from lib.loop import Buffer
from lib.model.lc0 import LC0Model
from lib.model.simple import SimpleNetwork
from lib.plotter import LogPlotter, start_qt_app
from lib.save_onnx import save_onnx
from lib.train import TrainSettings, WdlTarget, WdlLoss, batch_loader
from lib.util import DEVICE, print_param_count


def forward_hook(module, input, output):
    print("forward", module)


def backward_hook(module, grad_input, grad_output):
    print("backward", module)


def thread_main(logger: Logger, plotter: LogPlotter):
    data_folder = f"../../data/pgn-games-hist/cclr/test/"
    network_folder = "../../data/supervised/initial/"

    shutil.rmtree(network_folder, ignore_errors=True)
    os.makedirs(network_folder, exist_ok=True)

    game = Game.find("chess")
    paths = find_all_files(game, data_folder, pre_map=False)
    print("Paths found:")
    for p in paths:
        print(f"  {p}")

    batch_size = 256
    buffer_size = int(1e7)

    settings = TrainSettings(
        game=game,
        wdl_target=WdlTarget.Final,
        wdl_loss=WdlLoss.MSE,
        policy_weight=0.1,
        batch_size=batch_size,
        batches=32,
        clip_norm=100,
    )

    network = LC0Model(game, 64, 2, False)
    network.to(DEVICE)
    # network = SimpleNetwork(game, False)

    # for _, module in network.named_modules():
    #     module.register_forward_hook(forward_hook)
    #     module.register_full_backward_hook(backward_hook)

    print_param_count(network)

    # TODO weight decay?
    # TODO SDG vs Adam?

    # optimizer = SGD(network.parameters(), lr=0.001)
    # scheduler = CyclicLR(optimizer, 1e-4, 1e-1, step_size_up=1, step_size_down=100)
    optimizer = AdamW(network.parameters(), weight_decay=1e-5)

    buffer = Buffer(game, int(buffer_size), 0.05)

    path_iter = itertools.cycle(paths)

    # pre-fill buffer
    for path in path_iter:
        buffer.append(None, path)
        if len(buffer.full_train_dataset()) >= buffer_size:
            break

    for gi in itertools.count():
        print(f"Starting gen {gi}")
        logger.start_gen()

        # append a random buffer each time as a coarse form of shuffling
        # TODO this is super wrong, we're re-splitting train and test repeatedly and so we don't keep the separation properly
        buffer.append(logger, random.choice(paths))

        train_set = buffer.full_train_dataset()

        print(f"Buffer size: {len(train_set)}")

        settings.run_train(train_set, optimizer, network, logger, None)

        # evaluate the network
        network.eval()
        test_test_batch = next(iter(batch_loader(buffer.full_test_dataset(), batch_size))).to(DEVICE)
        test_train_batch = next(iter(batch_loader(buffer.full_train_dataset(), batch_size))).to(DEVICE)
        settings.evaluate_loss(network, "test-test", logger.log_gen, test_test_batch)
        settings.evaluate_loss(network, "test-train", logger.log_gen, test_train_batch)

        # compare to just predicting the mean value
        wdl = GameDataView(game, test_test_batch, includes_history=False).wdl_final
        value = wdl[:, 0] - wdl[:, 2]
        mean_value = value.mean()
        mean_value_loss = nnf.mse_loss(mean_value.expand(len(value)), value)
        logger.log_gen("loss-wdl", "trivial", mean_value_loss.item())

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


def map_single(args: Tuple[Game, str]):
    game, f = args
    GameDataFile(game, f).close()


def find_all_files(game: Game, folder: str, pre_map: bool):
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".bin.gz")]

    if pre_map:
        # TODO switch this back to multiprocessing once the "precess termination" issue with h5py is resolved
        pool = ThreadPool()
        args = [(game, p) for p in paths]
        pool.imap_unordered(map_single, args)

    return paths


if __name__ == '__main__':
    main()
