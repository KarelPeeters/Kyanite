import itertools
import os
import shutil
from threading import Thread

from torch.optim import AdamW, SGD

from lib.games import Game
from lib.logger import Logger
from lib.loop import Buffer
from lib.model import TowerModel, ResBlock
from lib.plotter import LogPlotter, start_qt_app
from lib.save_onnx import save_onnx
from lib.train import TrainSettings, WdlTarget, WdlLoss, batch_loader
from lib.util import DEVICE, print_param_count


def thread_main(logger: Logger, plotter: LogPlotter):
    network_folder = "../../data/supervised/initial/"
    shutil.rmtree(network_folder, ignore_errors=True)
    os.makedirs(network_folder, exist_ok=True)

    game = Game.find("chess")
    buffer = Buffer(game, int(1e15), 0.05)

    # TODO load more data at some point
    for i in range(1, 44):
        buffer.append(None, f"../../data/pgn-games/cclr/test/{i}.bin.gz")
    print(f"Buffer size: {len(buffer.full_train_dataset())}")

    batch_size = 1024
    test_loader = batch_loader(buffer.full_test_dataset(), batch_size)
    train_loader = batch_loader(buffer.full_test_dataset(), batch_size)

    settings = TrainSettings(
        game=game,
        wdl_target=WdlTarget.Final,
        wdl_loss=WdlLoss.MSE,
        policy_weight=50.0,
        batch_size=batch_size,
        batches=8,
        clip_norm=6.0,
    )

    def block():
        return ResBlock(game, 32, 32, True, False, False, None, False)

    network = TowerModel(game, 32, 8, 32, True, True, True, block)
    network.to(DEVICE)

    print_param_count(network)

    # TODO weight decay?
    # TODO SDG vs Adam?
    optimizer = SGD(network.parameters(), lr=1e-1, weight_decay=1e-5)

    for gi in itertools.count():
        logger.start_gen()

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
