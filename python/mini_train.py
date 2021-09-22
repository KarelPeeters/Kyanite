from threading import Thread

from torch.optim import AdamW

from lib.dataset import GameDataset
from lib.games import Game
from lib.logger import Logger
from lib.model import TowerModel, ResBlock
from lib.plotter import start_qt_app, LogPlotter
from lib.train import TrainSettings, WdlTarget, WdlLoss
from lib.util import DEVICE


def main():
    game = Game.find("chess")
    dataset = GameDataset.convert_and_open(game, "../data/var_game/tiny/selfplay/games_0.bin.gz")

    network = TowerModel(game, 32, 64, 32, True, True, True, lambda: ResBlock(32, 32, True, False, False, None))
    network.to(DEVICE)
    optimizer = AdamW(network.parameters(), weight_decay=1e-5)

    app = start_qt_app()
    logger = Logger()
    plotter = LogPlotter(logger)

    settings = TrainSettings(
        game=game,
        wdl_target=WdlTarget.Final,
        wdl_loss=WdlLoss.MSE,
        policy_weight=1.0,
        batch_size=128,
        batches=100,
    )

    def thread_main():
        logger.start_gen()
        settings.run_train(dataset, optimizer, network, logger)
        logger.finish_gen()
        plotter.update()

    Thread(target=thread_main).start()
    app.exec()


if __name__ == '__main__':
    main()
