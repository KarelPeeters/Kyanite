import itertools
import os
import time
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from typing import Callable, Optional, Tuple, Iterator, List

import torch
from torch import nn
from torch.optim import Optimizer

from lib.data.buffer import FileList
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.plotter import LogPlotter, qt_app
from lib.save_onnx import save_onnx
from lib.selfplay_client import SelfplaySettings, StartupSettings, SelfplayClient
from lib.train import TrainSettings
from lib.util import DEVICE, print_param_count
from main.write_test_networks import CHECK_BATCH_SIZE


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
    train_steps_per_gen: int

    # TODO re-implement testing
    # test_fraction: float
    # eval_steps_per_gen: int

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

        start_gen, buffer, logger, network, network_path_onnx = self.load_start_state()
        print_param_count(network)

        app = qt_app()
        plotter = LogPlotter()
        plotter.update(logger)

        # TODO this is a large a amount of tricky parameters, find a better way to pass them to the thread
        thread = Thread(target=self.run_loop_thread,
                        args=(start_gen, buffer, logger, plotter, network, network_path_onnx))
        thread.start()

        app.exec()

    def run_loop_thread(
            self,
            start_gen: 'Generation', buffer: 'LoopBuffer',
            logger: Logger, plotter: Optional[LogPlotter],
            network: nn.Module, network_path_onnx: str
    ):
        game = self.fixed_settings.game
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
            logger.start_batch()
            logger.log("info", "gen", gi)

            print(f"Waiting for gen {gi} games")
            gen_start = time.perf_counter()
            actual_gi = client.wait_for_file()
            client.send_wait_for_new_network()
            logger.log("time", "selfplay", time.perf_counter() - gen_start)
            assert gi == actual_gi, f"Unexpected finished generation, expected {gi} got {actual_gi}"

            gen = Generation.from_gi(self, gi)
            os.makedirs(gen.train_path, exist_ok=True)

            buffer.append(logger, DataFile(game, gen.games_path))
            self.evaluate_network(buffer, logger, network)

            train_dataset = buffer.full_file_list()
            print(f"Training network on buffer with size {len(train_dataset)}")
            train_start = time.perf_counter()

            for bi in range(self.train_steps_per_gen):
                if bi != 0:
                    logger.start_batch()

                self.train_settings.train_step(train_dataset, network, optimizer, logger)

            logger.log("time", "train", time.perf_counter() - train_start)

            torch.jit.save(network, gen.network_path_pt)
            save_onnx(game, gen.network_path_onnx, network, CHECK_BATCH_SIZE)
            client.send_new_network(gen.network_path_onnx)

            logger.save(self.log_path)
            Path(gen.finished_path).touch()

            if plotter is not None:
                plotter.update(logger)

    def load_start_state(self) -> Tuple['Generation', 'LoopBuffer', Logger, nn.Module, str]:
        game = self.fixed_settings.game
        buffer = LoopBuffer(game, self.target_buffer_size)

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
                    save_onnx(game, prev_network_path_onnx, network, CHECK_BATCH_SIZE)
                else:
                    print(f"Continuing run, first gen {gi}")
                    logger = Logger.load(self.log_path)

                    network = torch.jit.load(prev.network_path_pt)
                    network.to(DEVICE)

                    prev_network_path_onnx = prev.network_path_onnx

                return gen, buffer, logger, network, prev_network_path_onnx

            print(f"Found finished generation {gi}")
            buffer.append(None, DataFile(game, gen.games_path))

    def evaluate_network(self, buffer: 'LoopBuffer', logger: Logger, network: nn.Module):
        setups = [
            ("eval-test-buffer", buffer.full_file_list()),
            ("eval-test-last", buffer.last_file_list()),
        ]

        network.eval()
        for prefix, file_list in setups:
            batch_size = min(len(file_list), self.train_settings.batch_size)
            batch = file_list.sample_batch(batch_size)
            self.train_settings.evaluate_batch(network, prefix, logger, batch)


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
        games_path = os.path.join(settings.selfplay_path, f"games_{gi}")
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


class LoopBuffer:
    def __init__(self, game: Game, target_positions: int):
        self.game = game
        self.pool = ThreadPool(2)
        self.target_positions = target_positions

        self.current_positions = 0
        self.files: List[DataFile] = []

    def append(self, logger: Optional[Logger], file: DataFile):
        self.files.append(file)
        self.current_positions += len(file)

        while self.current_positions - len(self.files[0]) > self.target_positions:
            self.current_positions -= len(self.files[0])
            del self.files[0]

        if logger:
            total_games = sum(f.game_count for f in self.files)

            logger.log("buffer", "gens", len(self.files))
            logger.log("buffer", "games", total_games)
            logger.log("buffer", "positions", self.current_positions)

            logger.log("gen", "games", file.game_count)
            logger.log("gen", "positions", file.position_count)
            logger.log("gen", "game length min", file.min_game_length)
            logger.log("gen", "game length mean", file.position_count / file.game_count)
            logger.log("gen", "game length max", file.max_game_length)


    def full_file_list(self):
        return FileList(self.game, self.files, self.pool)

    def last_file_list(self):
        return FileList(self.game, [self.files[-1]], self.pool)
