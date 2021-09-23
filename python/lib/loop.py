import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Callable, Optional, Tuple, Iterator, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset

from lib.dataset import GameDataset, GameDataFile
from lib.games import Game
from lib.logger import Logger, FinishedLogData
from lib.plotter import LogPlotter, start_qt_app
from lib.save_onnx import save_onnx
from lib.selfplay_client import SelfplaySettings, StartupSettings, SelfplayClient
from lib.train import TrainSettings, batch_loader
from lib.util import DEVICE, print_param_count


@dataclass
class FixedSelfplaySettings:
    game: Game
    threads_per_device: int
    batch_size: int
    games_per_gen: int

    def to_startup(self, output_folder: str, first_gen: int):
        return StartupSettings(
            output_folder=output_folder,
            first_gen=first_gen,
            game=self.game.name,
            threads_per_device=self.threads_per_device,
            batch_size=self.batch_size,
            games_per_gen=self.games_per_gen,
        )


@dataclass
class LoopSettings:
    root_path: str
    initial_network: Callable[[], nn.Module]

    target_buffer_size: int
    test_fraction: float
    optimizer: Callable[[Iterator[nn.Parameter]], Optimizer]

    fixed_settings: FixedSelfplaySettings
    selfplay_settings: Optional[SelfplaySettings]
    train_settings: TrainSettings

    # TODO compact these properties somehow
    @property
    def initial_network_path_onnx(self):
        return os.path.join(self.root_path, "initial_network.onnx")

    @property
    def log_path(self):
        return os.path.join(self.root_path, "log.npz")

    @property
    def selfplay_path(self):
        return os.path.join(self.root_path, "selfplay")

    @property
    def training_path(self):
        return os.path.join(self.root_path, "training")

    def run_loop(self):
        print(f"Starting loop with cwd {os.getcwd()}")
        assert os.path.exists("./rust") and os.path.exists("./python"), "Should be run in root kZero folder"

        os.makedirs(self.selfplay_path, exist_ok=True)
        os.makedirs(self.training_path, exist_ok=True)

        # TODO this is a large a amount of tricky parameters, find a better way to pass them to the thread
        start_gen, buffer, logger, network, network_path_onnx = self.load_start_state()
        print_param_count(network)

        app = start_qt_app()
        plotter = LogPlotter(logger)
        if start_gen.gi != 0:
            plotter.update()

        thread = Thread(target=self.run_loop_thread,
                        args=(start_gen, buffer, logger, plotter, network, network_path_onnx))
        thread.start()

        app.exec()

    def run_loop_thread(
            self,
            start_gen: 'Generation', buffer: 'Buffer',
            logger: Logger, plotter: Optional[LogPlotter],
            network: nn.Module, network_path_onnx: str
    ):
        optimizer = self.optimizer(network.parameters())

        startup_settings = self.fixed_settings.to_startup(
            output_folder=self.selfplay_path,
            first_gen=start_gen.gi,
        )

        client = SelfplayClient()
        client.send_startup_settings(startup_settings)
        client.send_new_settings(self.selfplay_settings)
        client.send_new_network(network_path_onnx)

        for gi in itertools.count(start_gen.gi):
            logger.start_gen()

            print(f"Waiting for gen {gi} games")
            start = time.perf_counter()
            actual_gi = client.wait_for_file()
            client.send_wait_for_new_network()
            logger.log_gen("time", "selfplay", time.perf_counter() - start)
            assert gi == actual_gi, f"Unexpected finished generation, expected {gi} got {actual_gi}"

            gen = Generation.from_gi(self, gi)
            os.makedirs(gen.train_path, exist_ok=True)

            buffer.append(logger, gen.games_path)
            self.evaluate_network(buffer, logger, network)

            train_dataset = buffer.full_train_dataset()
            print(f"Training network on buffer with size {len(train_dataset)}")
            start = time.perf_counter()
            self.train_settings.run_train(train_dataset, optimizer, network, logger)
            logger.log_gen("time", "train", time.perf_counter() - start)

            torch.jit.save(network, gen.network_path_pt)
            save_onnx(self.fixed_settings.game, gen.network_path_onnx, network)
            client.send_new_network(os.path.abspath(gen.network_path_onnx))

            logger.finish_gen()
            logger.get_finished_data().save(self.log_path)

            Path(gen.finished_path).touch()

            if plotter is not None:
                plotter.update()

    def load_start_state(self) -> Tuple['Generation', 'Buffer', Logger, nn.Module, str]:
        game = self.fixed_settings.game
        buffer = Buffer(game, self.target_buffer_size, self.test_fraction)

        for gi in itertools.count():
            gen = Generation.from_gi(self, gi)
            prev = gen.prev

            if not os.path.exists(gen.finished_path):
                if prev is None:
                    print("Starting new run")
                    logger = Logger()

                    network = torch.jit.script(self.initial_network())
                    network.to(DEVICE)

                    prev_network_path_onnx = self.initial_network_path_onnx
                    save_onnx(game, prev_network_path_onnx, network)
                else:
                    print(f"Continuing run, first gen {gi}")
                    logger = Logger.from_finished_data(FinishedLogData.load(self.log_path))

                    network = torch.jit.load(prev.network_path_pt)
                    network.to(DEVICE)

                    prev_network_path_onnx = prev.network_path_onnx

                return gen, buffer, logger, network, prev_network_path_onnx

            print(f"Found finished generation {gi}")
            # TODO this splits train/test differently than the original one so plots may look strange
            buffer.append(None, gen.games_path)

    def evaluate_network(self, buffer: 'Buffer', logger: Logger, network: nn.Module):
        setups = [
            ("eval-test-buffer", buffer.full_test_dataset()),
            ("eval-test-last", buffer.last_test_dataset()),
            ("eval-train-buffer", buffer.full_train_dataset()),
            ("eval-train-last", buffer.last_train_dataset()),
        ]

        network.eval()
        for prefix, dataset in setups:
            # noinspection PyTypeChecker
            batch_size = min(len(dataset), self.train_settings.batch_size)
            batch = next(iter(batch_loader(dataset, batch_size))).to(DEVICE)
            self.train_settings.evaluate_loss(network, prefix, logger.log_gen, batch)


@dataclass
class Generation:
    settings: 'LoopSettings'
    gi: int
    games_path: str
    train_path: str
    network_path_pt: str
    network_path_onnx: str
    finished_path: str

    @classmethod
    def from_gi(cls, settings: 'LoopSettings', gi: int):
        games_path = os.path.join(settings.selfplay_path, f"games_{gi}.bin.gz")
        train_path = os.path.join(settings.training_path, f"gen_{gi}")

        return Generation(
            settings=settings,
            gi=gi,
            games_path=games_path,
            train_path=train_path,
            network_path_pt=os.path.join(train_path, "network.pt"),
            network_path_onnx=os.path.join(train_path, "network.onnx"),
            finished_path=os.path.join(train_path, "finished.txt"),
        )

    @property
    def prev(self):
        if self.gi == 0:
            return None
        return Generation.from_gi(self.settings, self.gi - 1)


class Buffer:
    def __init__(self, game: Game, target_size: int, test_fraction: float):
        self.game = game
        self.target_size = target_size
        self.test_fraction = test_fraction

        self.current_train: List[GameDataset] = []
        self.current_test: List[GameDataset] = []

    def append(self, logger: Optional[Logger], path: str):
        new = GameDataFile(self.game, path)

        new_train, new_test = new.split_dataset(self.test_fraction)
        self.current_train.append(new_train)
        self.current_test.append(new_test)

        # drop old datasets until we would go below the target size
        while sum([len(d) for d in self.current_train[1:]]) > self.target_size:
            del self.current_train[0]
            del self.current_test[0]

        new_full = new.full_dataset()
        if logger:
            logger.log_gen("game", "games/gen", new.game_count)
            logger.log_gen("game", "positions/game", len(new_full) / new.game_count)
            logger.log_gen("game", "shortest game", min(new.game_lengths))
            logger.log_gen("game", "longest game", max(new.game_lengths))

            logger.log_gen("buffer", "gens", len(self.current_train))
            logger.log_gen("buffer", "train positions", len(self.full_train_dataset()))
            logger.log_gen("buffer", "test positions", len(self.full_test_dataset()))
            logger.log_gen("buffer", "last test positions", len(self.last_test_dataset()))

    def full_train_dataset(self):
        return ConcatDataset(self.current_train)

    def last_train_dataset(self):
        return self.current_train[-1]

    def full_test_dataset(self):
        return ConcatDataset(self.current_test)

    def last_test_dataset(self):
        return self.current_test[-1]
