import glob
import itertools
import os
import shutil
from threading import Thread

import torch
import torch.nn.functional as nnf
from torch.optim import SGD

from lib.data.buffer import FileListSampler
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.model.lc0_pre_act import LCZOldPreNetwork
from lib.plotter import LogPlotter, qt_app
from lib.save_onnx import save_onnx
from lib.schedule import FixedSchedule, WarmupSchedule
from lib.train import TrainSettings
from lib.util import DEVICE, print_param_count


def thread_main(logger: Logger, plotter: LogPlotter):
    train_pattern = f"../../data/pgn/*.json"
    test_pattern = f"../../data/pgn/*.json"
    output_folder = "../../data/supervised/lichess_huge_lr/"

    shutil.rmtree(output_folder, ignore_errors=True)
    # assert not os.path.exists(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    game = Game.find("chess")

    batch_size = 1024

    test_steps = 16
    save_steps = 1028

    settings = TrainSettings(
        game=game,
        wdl_weight=0.5,
        value_weight=0.5,
        policy_weight=1.0,
        clip_norm=100,
    )

    network = LCZOldPreNetwork(game, 0, 16, 128, 8, 128)
    network.to(DEVICE)

    print_param_count(network)

    optimizer = SGD(network.parameters(), weight_decay=1e-5, lr=0.0)
    schedule = WarmupSchedule(100, FixedSchedule([0.03, 0.01, 0.001], [10_000, 1_000, 1_000]))

    train_files = [DataFile.open(game, path) for path in glob.glob(train_pattern)]
    train_sampler = FileListSampler(game, train_files, batch_size)

    test_files = [DataFile.open(game, path) for path in glob.glob(test_pattern)]
    test_sampler = FileListSampler(game, test_files, batch_size)

    print(f"Train file count: {len(train_files)}")
    print(f"Train position count: {len(train_sampler)}")

    print(f"Test file count: {len(test_files)}")
    print(f"Test position count: {len(test_sampler)}")

    for bi in itertools.count():
        print(f"Starting batch {bi}")
        logger.start_batch()

        if schedule is not None:
            lr = schedule(bi)
            logger.log("schedule", "lr", lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

        settings.train_step(train_sampler, network, optimizer, logger)

        if bi % test_steps == 0:
            network.eval()

            train_batch = train_sampler.next_batch()
            settings.evaluate_batch(network, "test-train", logger, train_batch)

            test_batch = test_sampler.next_batch()
            settings.evaluate_batch(network, "test-test", logger, test_batch)

            # compare to just predicting the mean value
            mean_wdl = train_batch.wdl_final.mean(axis=0, keepdims=True)
            mean_value = mean_wdl[0, 0] - mean_wdl[0, 2]

            loss_wdl_mean = nnf.mse_loss(mean_wdl.expand(len(test_batch), 3), test_batch.wdl_final)

            test_batch_final_value = test_batch.wdl_final[:, 0] - test_batch.wdl_final[:, 2]
            loss_value_mean = nnf.mse_loss(mean_value.expand(len(test_batch)), test_batch_final_value)

            logger.log("loss-wdl", "trivial", loss_wdl_mean.item())
            logger.log("loss-value", "trivial", loss_value_mean.item())

            plotter.update(logger)

            print("Saving log")
            logger.save(os.path.join(output_folder, "log.npz"))

        if bi % save_steps == 0:
            print("Saving network")
            save_onnx(game, os.path.join(output_folder, f"network_{bi}.onnx"), network, None)
            torch.jit.script(network).save(os.path.join(output_folder, f"network_{bi}.pb"))

    train_sampler.close()
    test_sampler.close()


def main():
    app = qt_app()

    logger = Logger()
    plotter = LogPlotter()

    thread = Thread(target=thread_main, args=(logger, plotter), daemon=True)
    thread.start()

    app.exec()


if __name__ == '__main__':
    main()
