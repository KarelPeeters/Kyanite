import itertools
import os
import time
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Optional, Tuple, Iterator, List

import torch
from torch import nn
from torch.optim import Optimizer

from lib.data.buffer import FileListSampler
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.plotter import LogPlotter, run_with_plotter
from lib.save_onnx import save_onnx
from lib.selfplay_client import SelfplaySettings, StartupSettings, SelfplayClient
from lib.train import TrainSettings
from lib.util import DEVICE, print_param_count

CHECK_BATCH_SIZE = 2
SAVE_BATCH_SIZE = 2


@dataclass
class FixedSelfplaySettings:
    game: Game
    threads_per_device: int
    batch_size: int
    games_per_gen: int
    reorder_games: bool

    def to_startup(self, output_folder: str, first_gen: int):
        return StartupSettings(
            output_folder=os.path.abspath(output_folder),
            first_gen=first_gen,
            game=self.game.name,
            threads_per_device=self.threads_per_device,
            batch_size=self.batch_size,
            games_per_gen=self.games_per_gen,
            reorder_games=self.reorder_games,
        )


@dataclass
class LoopSettings:
    gui: bool
    root_path: str

    dummy_network: Optional[Callable[[], nn.Module]]
    initial_network: Callable[[], nn.Module]
    initial_data_files: List[DataFile]

    only_generate: bool

    min_buffer_size: int
    max_buffer_size: int

    train_batch_size: int
    samples_per_position: float

    # TODO re-implement testing
    # test_fraction: float
    # eval_steps_per_gen: int

    optimizer: Callable[[Iterator[nn.Parameter]], Optimizer]

    fixed_settings: FixedSelfplaySettings
    selfplay_settings: Optional[SelfplaySettings]
    train_settings: TrainSettings

    # TODO compact these properties somehow
    @property
    def dummy_network_path_onnx(self):
        return os.path.join(self.root_path, "dummy_network.onnx")

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

    def calc_batch_count_per_gen(self) -> int:
        game = self.fixed_settings.game
        positions_in_buffer = self.max_buffer_size

        # this does not depend on gens_in_buffer since that divides itself away
        positions_per_gen = game.estimate_moves_per_game * self.fixed_settings.games_per_gen
        batch_count = self.samples_per_position * positions_per_gen / self.train_batch_size

        batch_count_int = round(batch_count)
        if batch_count_int == 0:
            batch_count_int = 1

        # extra calculations for prints
        gens_in_buffer = positions_in_buffer / positions_per_gen
        games_in_buffer = gens_in_buffer * self.fixed_settings.games_per_gen
        samples_per_game = self.samples_per_position * game.estimate_moves_per_game
        samples_per_gen = samples_per_game * self.fixed_settings.games_per_gen

        print("Behaviour estimates:")
        print(f"  Gen:")
        print(f"    {self.fixed_settings.games_per_gen} games")
        print(f"    {positions_per_gen} positions")
        print(f"  Buffer:")
        print(f"    {gens_in_buffer:.4} gens")
        print(f"    {games_in_buffer:.4} games")
        print(f"    {positions_in_buffer} positions")
        print(f"  Sampling rate:")
        print(f"    {samples_per_gen:.4} /gen")
        print(f"    {samples_per_game:.4} /game")
        print(f"    {self.samples_per_position:.4} /position")
        print(f"Calculated {batch_count:.4} -> {batch_count_int} batches per gen")

        return batch_count_int

    def run_loop(self):
        print(f"Starting loop with cwd {os.getcwd()}")
        assert os.path.exists("./rust") and os.path.exists("./python"), \
            f"Should be run in root kZero folder, got {os.getcwd()}"

        os.makedirs(self.selfplay_path, exist_ok=True)
        os.makedirs(self.training_path, exist_ok=True)

        start_gen, buffer, logger, network, network_path_onnx = self.load_start_state()
        print_param_count(network)

        def target(plotter: Optional[LogPlotter]):
            if plotter is not None:
                # TODO why does this crash the plotter?
                # plotter.set_title(f"loop: {self.root_path}")
                plotter.set_can_pause(False)

            self.run_loop_inner(start_gen, buffer, logger, plotter, network, network_path_onnx)

        if self.gui:
            run_with_plotter(target)
        else:
            target(None)

    def run_loop_inner(
            self,
            start_gen: 'Generation', buffer: 'LoopBuffer',
            logger: Logger, plotter: Optional[LogPlotter],
            network: nn.Module, network_path_onnx: str
    ):
        game = self.fixed_settings.game
        optimizer = self.optimizer(network.parameters())
        batch_count_per_gen = self.calc_batch_count_per_gen()

        startup_settings = self.fixed_settings.to_startup(
            output_folder=self.selfplay_path,
            first_gen=start_gen.gi,
        )

        client = SelfplayClient()
        client.send_startup_settings(startup_settings)
        client.send_new_settings(self.selfplay_settings)

        if self.dummy_network is not None and network_path_onnx == self.initial_network_path_onnx:
            client.send_new_network(self.dummy_network_path_onnx)
        else:
            client.send_new_network(network_path_onnx)

        for gi in itertools.count(start_gen.gi):
            if plotter is not None:
                plotter.update(logger)
                plotter.block_while_paused()

            logger.start_batch()
            logger.log("info", "gen", gi)

            print(f"Waiting for gen {gi} games")
            gen_start = time.perf_counter()
            actual_gi = client.wait_for_file()
            logger.log("time", "selfplay", time.perf_counter() - gen_start)
            assert gi == actual_gi, f"Unexpected finished generation, expected {gi} got {actual_gi}"

            if self.only_generate:
                print("Not training new network, we're only generating data")
                continue

            gen = Generation.from_gi(self, gi)
            os.makedirs(gen.train_path, exist_ok=True)

            buffer.append(logger, DataFile.open(game, gen.games_path))

            if buffer.position_count < self.min_buffer_size:
                print(
                    f"Not training new network yet, only got {buffer.position_count}/{self.min_buffer_size} positions")
            else:
                client.send_wait_for_new_network()
                self.evaluate_network(buffer, logger, network)

                train_sampler = buffer.sampler_full(self.train_batch_size)
                print(f"Training network on buffer with size {len(train_sampler)}")
                train_start = time.perf_counter()

                for bi in range(batch_count_per_gen):
                    if bi != 0:
                        logger.start_batch()

                    train_batch = train_sampler.next_batch()
                    self.train_settings.train_step(train_batch, network, optimizer, logger)
                train_sampler.close()

                logger.log("time", "train", time.perf_counter() - train_start)

                torch.jit.save(network, gen.network_path_pt)
                save_onnx(game, gen.network_path_onnx, network, 2)
                client.send_new_network(gen.network_path_onnx)

            logger.save(self.log_path)
            Path(gen.finished_path).touch()

    def load_start_state(self) -> Tuple['Generation', 'LoopBuffer', Logger, nn.Module, str]:
        game = self.fixed_settings.game
        buffer = LoopBuffer(game, self.max_buffer_size)

        for file in self.initial_data_files:
            buffer.append(None, file)
        print(f"Initial buffer: {len(buffer.files)} files, {buffer.game_count} games, {buffer.position_count}")

        for gi in itertools.count():
            gen = Generation.from_gi(self, gi)
            prev = gen.prev

            if not os.path.exists(gen.finished_path):
                if prev is None:
                    print("Starting new run")
                    logger = Logger()

                    # test if the network actually even works
                    network_raw = self.initial_network()
                    network_raw.to(DEVICE)
                    network_raw(torch.randn(CHECK_BATCH_SIZE, *game.full_input_shape, device=DEVICE))
                    torch.cuda.synchronize()

                    # convert and save the network
                    network = torch.jit.script(network_raw)
                    network.to(DEVICE)

                    prev_network_path_onnx = self.initial_network_path_onnx
                    save_onnx(game, prev_network_path_onnx, network, SAVE_BATCH_SIZE)

                    # save the dummy network too if there is one
                    if self.dummy_network is not None:
                        dummy_network = self.dummy_network()
                        save_onnx(game, self.dummy_network_path_onnx, dummy_network, SAVE_BATCH_SIZE)
                else:
                    print(f"Continuing run, first gen {gi}")
                    logger = Logger.load(self.log_path)

                    network = torch.jit.load(prev.network_path_pt)
                    network.to(DEVICE)

                    prev_network_path_onnx = prev.network_path_onnx

                return gen, buffer, logger, network, prev_network_path_onnx

            print(f"Found finished generation {gi}")
            try:
                buffer.append(None, DataFile.open(game, gen.games_path))
            except FileNotFoundError:
                print(f"Could not find games file for gen {gi}, skipping")

    def evaluate_network(self, buffer: 'LoopBuffer', logger: Logger, network: nn.Module):
        setups = [
            ("eval-test-buffer", buffer.sampler_full(self.train_batch_size)),
            ("eval-test-last", buffer.sampler_last(self.train_batch_size)),
        ]

        network.eval()
        for prefix, sampler in setups:
            batch = sampler.next_batch()
            self.train_settings.evaluate_batch(network, batch, prefix, logger)
            sampler.close()


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

        self.position_count = 0
        self.game_count = 0
        self.files: List[DataFile] = []

    def append(self, logger: Optional[Logger], file: DataFile):
        assert file.info.game == self.game, f"Expected game {self.game.name}, got game {file.info.game.name}"

        self.files.append(file)
        self.position_count += len(file)
        self.game_count += file.info.game_count

        while self.position_count - len(self.files[0]) > self.target_positions:
            old_file = self.files[0]
            del self.files[0]

            self.position_count -= len(old_file)
            self.game_count -= old_file.info.game_count
            old_file.close()

        if logger:
            logger.log("buffer", "gens", len(self.files))
            logger.log("buffer", "games", self.game_count)
            logger.log("buffer", "positions", self.position_count)

            info = file.info

            logger.log("gen-size", "games", info.game_count)
            logger.log("gen-size", "positions", info.position_count)
            logger.log("gen-game-len", "game length min", info.min_game_length)
            logger.log("gen-game-len", "game length mean", info.position_count / info.game_count)
            logger.log("gen-game-len", "game length max", info.max_game_length)

            if info.root_wdl is not None:
                logger.log("gen-root-wdl", "w", info.root_wdl[0])
                logger.log("gen-root-wdl", "d", info.root_wdl[1])
                logger.log("gen-root-wdl", "l", info.root_wdl[2])

    def sampler_full(self, batch_size: int):
        return FileListSampler(self.game, self.files, batch_size, unroll_steps=None, threads=1)

    def sampler_last(self, batch_size: int):
        return FileListSampler(self.game, [self.files[-1]], batch_size, unroll_steps=None, threads=1)
