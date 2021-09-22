import time

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from benchmark_limit import bench_limit
from lib.dataset import GameDataset
from lib.games import Game
from lib.model import TowerModel, ResBlock
from train.module import TrainModule


def main():
    game = Game.find("chess")

    model = TowerModel(game, 32, 64, 32, True, True, True, lambda: ResBlock(32, 32, True, False, False, None))
    module = TrainModule(game, model)

    dataset = GameDataset.convert_and_open(game, "../data/var_game/tiny/selfplay/games_0.bin.gz")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, pin_memory=True)

    start = time.perf_counter()
    bench_limit(module, loader, 100)
    print(f"Bench took {time.perf_counter() - start}s")

    ignore_warnings()
    trainer = Trainer(
        max_steps=100,
        gpus=1,
        weights_summary="full",
        log_every_n_steps=2, checkpoint_callback=False,
    )

    start = time.perf_counter()
    trainer.fit(module, loader)
    print(f"Trainer took {time.perf_counter() - start}s")


def ignore_warnings():
    import warnings
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument`.*")


if __name__ == '__main__':
    main()
