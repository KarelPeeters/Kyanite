import itertools
import os
import random
import shutil
from multiprocessing.pool import Pool
from threading import Thread
from typing import Tuple

import torch
import torch.nn.functional as nnf
from torch.optim import AdamW

from experimental.grad_norms import plot_grad_norms
from lib.dataset import GameDataFile
from lib.dataview import GameDataView
from lib.games import Game
from lib.logger import Logger
from lib.loop import Buffer
from lib.model.lc0 import LC0Model
from lib.model.lc0_fixup import LC0FixupModel
from lib.plotter import LogPlotter, start_qt_app
from lib.save_onnx import save_onnx
from lib.train import TrainSettings, batch_loader
from lib.util import DEVICE, print_param_count

#TODO note:
#  how can value even possibly overfit? the only real difference is the BN layers!
#  in plot_act for gen 200 (after some value overwriting) the value dense layer output barely changes,
#    even for completely different boards!
#  try removing move count and even the color planes!

def thread_main(logger: Logger, plotter: LogPlotter):
    data_folder = f"../../data/pgn-games/cclr/test/"
    network_folder = "../../data/supervised/initial/"

    shutil.rmtree(network_folder, ignore_errors=True)
    os.makedirs(network_folder, exist_ok=True)

    game = Game.find("chess")
    paths = find_all_files(game, data_folder, pre_map=False)
    print("Paths found:")
    for p in paths:
        print(f"  {p}")

    batch_size = 512
    buffer_size = int(1e6)

    settings = TrainSettings(
        game=game,
        policy_weight=1.0,
        batch_size=batch_size,
        batches=32,
        clip_norm=100,
    )

    network = LC0Model(game, 32, 4, True)
    # network = LC0FixupModel(game, True, 64)
    network.to(DEVICE)
    # network = SimpleNetwork(game, False)

    print_param_count(network)

    # TODO weight decay?
    # TODO SDG vs Adam?

    # optimizer = SGD(network.parameters(), lr=0.2)
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
        mean_wdl = wdl.mean(dim=0, keepdims=True)
        loss_wdl_mean = nnf.mse_loss(mean_wdl.expand(len(wdl), 3), wdl)
        logger.log_gen("loss-wdl", "trivial", loss_wdl_mean.item())

        logger.finish_gen()
        plotter.update()

        if gi % 100 == 0:
            # plot_grad_norms(settings, network, test_test_batch)
            save_onnx(game, os.path.join(network_folder, f"network_{gi}.onnx"), network)
            torch.jit.script(network).save(os.path.join(network_folder, f"network_{gi}.pb"))


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
        pool = Pool()
        args = [(game, p) for p in paths]
        pool.imap_unordered(map_single, args)

    return paths


if __name__ == '__main__':
    main()
