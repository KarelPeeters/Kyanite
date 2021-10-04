import glob
import itertools
import os
import shutil
from multiprocessing.pool import ThreadPool
from threading import Thread

import torch
import torch.nn.functional as nnf
from torch.optim import SGD

from lib.data.buffer import FileBuffer
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.model.lc0_old import LCZOldNetwork
from lib.plotter import LogPlotter, qt_app
from lib.save_onnx import save_onnx
from lib.schedule import Schedule
from lib.train import TrainSettings
from lib.util import DEVICE, print_param_count


def thread_main(logger: Logger, plotter: LogPlotter):
    train_pattern = f"../../data/pgn-games/ccrl/train/*.json"
    test_pattern = f"../../data/pgn-games/ccrl/test/*.json"
    output_folder = "../../data/supervised/derp2/"

    shutil.rmtree(output_folder, ignore_errors=True)
    # assert not os.path.exists(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    game = Game.find("chess")

    batch_size = 128
    test_steps = 16
    save_steps = 1000

    settings = TrainSettings(
        game=game,
        wdl_weight=0.1,
        value_weight=1.0,
        policy_weight=1.0,
        batch_size=batch_size,
        clip_norm=100,
    )

    network = LCZOldNetwork(game, 32, 4)
    network.to(DEVICE)

    print_param_count(network)

    optimizer = SGD(network.parameters(), weight_decay=1e-5, lr=0.1)
    schedule = Schedule(20, [0.1, 0.01, 0.001], [100, 200])

    # optimizer = AdamW(network.parameters(), weight_decay=1e-5)
    # scheduler = None

    pool = ThreadPool(4)

    train_files = [DataFile(game, path) for path in glob.glob(train_pattern)]
    train_buffer = FileBuffer(game, train_files, pool)

    test_files = [DataFile(game, path) for path in glob.glob(test_pattern)]
    test_buffer = FileBuffer(game, test_files, pool)

    print(f"Train file count: {len(train_files)}")
    print(f"Train position count: {len(train_buffer)}")

    print(f"Test file count: {len(test_files)}")
    print(f"Test position count: {len(test_buffer)}")

    for bi in itertools.count():
        logger.start_batch()

        if schedule is not None:
            lr = schedule.get(bi)
            logger.log("schedule", "lr", lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

        settings.train_step(train_buffer, network, optimizer, logger)

        if bi % test_steps == 0:
            network.eval()

            train_batch = train_buffer.sample_batch(batch_size)
            settings.evaluate_batch(network, "test-train", logger, train_batch)

            test_batch = test_buffer.sample_batch(batch_size)
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

        if bi % save_steps == 0:
            # plot_grad_norms(settings, network, test_test_batch)
            save_onnx(game, os.path.join(output_folder, f"network_{bi}.onnx"), network)
            torch.jit.script(network).save(os.path.join(output_folder, f"network_{bi}.pb"))

        tmp_log_path = os.path.join(output_folder, "log.tmp.npz")
        log_path = os.path.join(output_folder, "log.npz")
        logger.save(tmp_log_path)
        os.replace(tmp_log_path, log_path)


def main():
    app = qt_app()

    logger = Logger()
    plotter = LogPlotter()

    thread = Thread(target=thread_main, args=(logger, plotter), daemon=True)
    thread.start()

    app.exec()


if __name__ == '__main__':
    main()
