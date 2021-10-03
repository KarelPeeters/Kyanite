import glob
import itertools
import os
import shutil
from multiprocessing.pool import ThreadPool
from threading import Thread

import torch
import torch.nn.functional as nnf
from torch.optim import AdamW

from lib.data.buffer import FileBuffer
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.model.lc0_old import LCZOldNetwork
from lib.model.lc0_pre_act import LCZOldPreNetwork
from lib.plotter import LogPlotter, start_qt_app
from lib.save_onnx import save_onnx
from lib.train import TrainSettings
from lib.util import DEVICE, print_param_count


def thread_main(logger: Logger, plotter: LogPlotter):
    train_pattern = f"../../data/pgn-games/ccrl/train/*.json"
    test_pattern = f"../../data/pgn-games/ccrl/test/*.json"
    output_folder = "../../data/supervised/fixed_out/"

    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)

    game = Game.find("chess")

    batch_size = 128

    settings = TrainSettings(
        game=game,
        wdl_weight=0.1,
        value_weight=1.0,
        policy_weight=1.0,
        batch_size=batch_size,
        batches=16,
        clip_norm=100,
    )

    network = LCZOldPreNetwork(game, 128, 8)
    # network = ConstantNetwork(game)
    network.to(DEVICE)

    print_param_count(network)

    # optimizer = SGD(network.parameters(), weight_decay=1e-5, lr=0.01)
    # scheduler = MultiStepLR(optimizer, [200*32, 400*32, 600*32], gamma=0.1)

    optimizer = AdamW(network.parameters(), weight_decay=1e-5)
    scheduler = None

    pool = ThreadPool(4)

    train_files = [DataFile(game, path) for path in glob.glob(train_pattern)]
    train_buffer = FileBuffer(game, train_files, pool)

    test_files = [DataFile(game, path) for path in glob.glob(test_pattern)]
    test_buffer = FileBuffer(game, test_files, pool)

    print(f"Train file count: {len(train_files)}")
    print(f"Train position count: {len(train_buffer)}")

    print(f"Test file count: {len(test_files)}")
    print(f"Test position count: {len(test_buffer)}")

    for gi in itertools.count():
        print(f"Starting gen {gi}")

        logger.start_gen()

        settings.run_train(train_buffer, optimizer, network, logger, scheduler)

        # evaluate the network
        network.eval()

        train_batch = train_buffer.sample_batch(batch_size)
        settings.evaluate_batch(network, "test-train", logger.log_gen, train_batch)

        test_batch = test_buffer.sample_batch(batch_size)
        settings.evaluate_batch(network, "test-test", logger.log_gen, test_batch)

        # compare to just predicting the mean value
        mean_wdl = train_batch.wdl_final.mean(axis=0, keepdims=True)
        mean_value = mean_wdl[0, 0] - mean_wdl[0, 2]

        loss_wdl_mean = nnf.mse_loss(mean_wdl.expand(len(test_batch), 3), test_batch.wdl_final)

        test_batch_final_value = test_batch.wdl_final[:, 0] - test_batch.wdl_final[:, 2]
        loss_value_mean = nnf.mse_loss(mean_value.expand(len(test_batch)), test_batch_final_value)

        logger.log_gen("loss-wdl", "trivial", loss_wdl_mean.item())
        logger.log_gen("loss-value", "trivial", loss_value_mean.item())

        logger.finish_gen()
        plotter.update()

        if gi % 100 == 0:
            # plot_grad_norms(settings, network, test_test_batch)
            save_onnx(game, os.path.join(output_folder, f"network_{gi}.onnx"), network)
            torch.jit.script(network).save(os.path.join(output_folder, f"network_{gi}.pb"))

        tmp_log_path = os.path.join(output_folder, "log.tmp.npz")
        log_path = os.path.join(output_folder, "log.npz")
        logger.get_finished_data().save(tmp_log_path)
        os.replace(tmp_log_path, log_path)


def main():
    logger = Logger()
    plotter = LogPlotter(logger)

    app = start_qt_app()
    thread = Thread(target=thread_main, args=(logger, plotter), daemon=True)
    thread.start()
    app.exec()


if __name__ == '__main__':
    main()
