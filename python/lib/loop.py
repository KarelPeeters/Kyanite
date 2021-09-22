import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Callable, Optional, Tuple, Iterator

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset

from lib.dataset import GameDataset
from lib.games import Game
from lib.logger import Logger, FinishedLogData
from lib.plotter import LogPlotter, start_qt_app
from lib.save_onnx import save_onnx
from lib.selfplay_client import SelfplaySettings, StartupSettings, SelfplayClient
from lib.train import TrainSettings
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
            buffer_dataset = buffer.as_dataset()

            start = time.perf_counter()
            print(f"Training network on buffer with size {len(buffer_dataset)}")
            self.train_settings.run_train(buffer_dataset, optimizer, network, logger)
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
        buffer = Buffer(game, self.target_buffer_size)

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
            buffer.append(None, gen.games_path)


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
    def __init__(self, game: Game, target_size: int):
        self.game = game
        self.target_size = target_size
        self.current = []

    def append(self, logger: Optional[Logger], path: str):
        new = GameDataset.convert_and_open(self.game, path)
        self.current.append(new)

        # drop old datasets until we would go below the target size
        while sum([len(d) for d in self.current[1:]]) > self.target_size:
            del self.current[0]

        if logger:
            logger.log_gen("game", "games/gen", new.game_count)
            logger.log_gen("game", "positions/game", len(new) / new.game_count)
            logger.log_gen("game", "shortest game", min(new.game_lengths))
            logger.log_gen("game", "longest game", max(new.game_lengths))

            logger.log_gen("buffer", "gens", len(self.current))
            logger.log_gen("buffer", "positions", len(self.as_dataset()))

    def as_dataset(self):
        return ConcatDataset(self.current)
