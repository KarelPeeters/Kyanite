import itertools
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, Iterator, List

import torch
from torch import nn
from torch.optim import Optimizer

from lib.data.file import DataFile
from lib.data.group import DataGroup
from lib.data.sampler import PositionSampler
from lib.games import Game
from lib.logger import Logger
from lib.plotter import LogPlotter, run_with_plotter
from lib.save_onnx import save_onnx, save_muzero_onnx
from lib.selfplay_client import SelfplaySettings, StartupSettings, SelfplayClient
from lib.train import TrainSettings
from lib.util import DEVICE, print_param_count, clean_folder

CHECK_BATCH_SIZE = 2
SAVE_BATCH_SIZE = 2


@dataclass
class FixedSelfplaySettings:
    game: Game
    muzero: bool
    start_pos: str

    simulations_per_gen: int

    cpu_threads_per_device: int
    gpu_threads_per_device: int
    gpu_batch_size: int
    gpu_batch_size_root: int
    search_batch_size: int

    saved_state_channels: int

    def to_startup(self, output_folder: str, first_gen: int):
        return StartupSettings(
            game=self.game.name,
            muzero=self.muzero,
            start_pos=self.start_pos,
            first_gen=first_gen,
            output_folder=os.path.abspath(output_folder),
            games_per_gen=self.simulations_per_gen,
            cpu_threads_per_device=self.cpu_threads_per_device,
            gpu_threads_per_device=self.gpu_threads_per_device,
            gpu_batch_size=self.gpu_batch_size,
            gpu_batch_size_root=self.gpu_batch_size_root,
            search_batch_size=self.search_batch_size,
            saved_state_channels=self.saved_state_channels,
        )


@dataclass
class LoopSettings:
    gui: bool
    root_path: str
    port: int
    wait_for_new_network: bool

    dummy_network: Optional[Callable[[], nn.Module]]
    initial_network: Callable[[], nn.Module]
    initial_data_files: List[DataFile]

    only_generate: bool

    min_buffer_size: int
    max_buffer_size: int

    train_batch_size: int
    samples_per_position: float
    test_fraction: float

    # TODO re-implement testing
    # test_fraction: float
    # eval_steps_per_gen: int

    optimizer: Callable[[Iterator[nn.Parameter]], Optimizer]

    fixed_settings: FixedSelfplaySettings
    selfplay_settings: Optional[SelfplaySettings]
    train_settings: TrainSettings

    muzero_steps: Optional[int]
    include_final: bool

    muzero: bool = field(init=False)

    log_path: str = field(init=False)
    selfplay_path: str = field(init=False)
    training_path: str = field(init=False)
    tmp_path: str = field(init=False)

    def __post_init__(self):
        self.muzero = self.muzero_steps is not None
        assert self.muzero == self.fixed_settings.muzero, f"Muzero state mismatch, got steps {self.muzero_steps} but fixed settings {self.fixed_settings.muzero}"

        self.log_path = os.path.join(self.root_path, "log.npz")
        self.selfplay_path = os.path.join(self.root_path, "selfplay")
        self.training_path = os.path.join(self.root_path, "training")
        self.tmp_path = os.path.join(self.root_path, "tmp")

    def calc_batch_count_per_gen(self) -> int:
        game = self.fixed_settings.game
        positions_in_buffer = self.max_buffer_size
        samples_per_position = float(self.samples_per_position)

        # this does not depend on gens_in_buffer since that divides itself away
        positions_per_gen = game.estimate_moves_per_game * self.fixed_settings.simulations_per_gen

        samples_per_batch = self.train_batch_size * (1 + (self.muzero_steps or 0))
        batch_count = samples_per_position * positions_per_gen / samples_per_batch

        batch_count_int = round(batch_count)
        if batch_count_int == 0:
            batch_count_int = 1

        # extra calculations for prints
        gens_in_buffer = positions_in_buffer / positions_per_gen
        simulations_in_buffer = gens_in_buffer * self.fixed_settings.simulations_per_gen
        samples_per_game = samples_per_position * game.estimate_moves_per_game
        samples_per_gen = samples_per_game * self.fixed_settings.simulations_per_gen

        print("Behaviour estimates:")
        print(f"  Gen:")
        print(f"    {self.fixed_settings.simulations_per_gen} games")
        print(f"    {positions_per_gen} positions")
        print(f"  Buffer:")
        print(f"    {gens_in_buffer:.4} gens")
        print(f"    {simulations_in_buffer:.4} games")
        print(f"    {positions_in_buffer} positions")
        print(f"  Sampling rate:")
        print(f"    {samples_per_batch} /batch")
        print(f"    {samples_per_gen:.4} /gen")
        print(f"    {samples_per_game:.4} /game")
        print(f"    {samples_per_position :.4} /position")
        print(f"Calculated {batch_count:.4} -> {batch_count_int} batches per gen")

        return batch_count_int

    def run_loop(self):
        print(f"Starting loop with cwd {os.getcwd()}")
        assert os.path.exists("./rust") and os.path.exists("./python"), \
            f"Should be run in root kZero folder, got {os.getcwd()}"

        os.makedirs(self.selfplay_path, exist_ok=True)
        os.makedirs(self.training_path, exist_ok=True)
        clean_folder(self.tmp_path)

        start_gen, buffer, logger, network = self.load_start_state()

        def target(plotter: Optional[LogPlotter]):
            if plotter is not None:
                plotter.set_title(f"loop: {self.root_path}")
                plotter.set_can_pause(False)

            self.run_loop_inner(start_gen, buffer, logger, plotter, network)

        if self.gui:
            run_with_plotter(target)
        else:
            target(None)

    def run_loop_inner(
            self,
            start_gen: 'Generation', buffer: 'LoopBuffer',
            logger: Logger, plotter: Optional[LogPlotter],
            network: nn.Module,
    ):
        print("Saving current settings")
        os.makedirs(start_gen.train_path, exist_ok=True)
        with open(start_gen.settings_path, "w") as settings_f:
            def map(o):
                if isinstance(o, range):
                    return str(o)
                return o.__dict__

            json.dump(self, settings_f, default=map, indent=2)

        print("Main network parameters:")
        print_param_count(network)

        game = self.fixed_settings.game
        optimizer = self.optimizer(network.parameters())
        batch_count_per_gen = self.calc_batch_count_per_gen()

        startup_settings = self.fixed_settings.to_startup(
            output_folder=self.selfplay_path,
            first_gen=start_gen.gi,
        )

        if buffer.position_count < self.min_buffer_size:
            if self.dummy_network is None:
                initial_onnx_path = self.save_tmp_onnx_network(network, "network_initial")
            else:
                dummy_network = torch.jit.script(self.dummy_network())
                print("Dummy network parameters:")
                print_param_count(dummy_network)

                initial_onnx_path = self.save_tmp_onnx_network(dummy_network, "network_dummy")
        else:
            initial_onnx_path = self.save_tmp_onnx_network(network, f"network_{start_gen.gi}")

        client = SelfplayClient(self.port)
        client.send_startup_settings(startup_settings)
        client.send_new_settings(self.selfplay_settings)
        client.send_new_network(initial_onnx_path)

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

            buffer.append(logger, DataFile.open(game, gen.simulations_path))

            if buffer.position_count < self.min_buffer_size:
                print(f"Not training new network yet, only {buffer.position_count}/{self.min_buffer_size} positions")
            else:
                if self.wait_for_new_network:
                    client.send_wait_for_new_network()
                self.evaluate_network(buffer, logger, network)

                train_sampler = buffer.sampler(
                    self.train_batch_size, self.muzero_steps, self.include_final,
                    only_last=False, test=False
                )
                print(
                    f"Training network on buffer with size {len(train_sampler.group.positions)} for {batch_count_per_gen} batches")
                train_start = time.perf_counter()

                for bi in range(batch_count_per_gen):
                    if bi != 0:
                        logger.start_batch()

                    train_batch = train_sampler.next_batch_either()
                    self.train_settings.train_step(train_batch, network, optimizer, logger)
                train_sampler.close()

                logger.log("time", "train", time.perf_counter() - train_start)

                torch.jit.save(network, gen.network_path_pt)

                curr_onnx_path = self.save_tmp_onnx_network(network, f"network_{gen.gi}")
                client.send_new_network(curr_onnx_path)

            logger.save(self.log_path)
            Path(gen.finished_path).touch()

    def load_start_state(self) -> Tuple['Generation', 'LoopBuffer', Logger, nn.Module]:
        game = self.fixed_settings.game
        buffer = LoopBuffer(game, self.max_buffer_size, self.test_fraction)

        for file in self.initial_data_files:
            buffer.append(None, file)
        print(f"Initial buffer: {len(buffer.files)} files, {buffer.simulation_count} games, {buffer.position_count}")

        for gi in itertools.count():
            gen = Generation.from_gi(self, gi)
            prev = gen.prev

            if os.path.exists(gen.finished_path):
                print(f"Found finished generation {gi}")
                try:
                    buffer.append(None, DataFile.open(game, gen.simulations_path))
                except FileNotFoundError:
                    print(f"Could not find games file for gen {gi}, skipping")
            else:
                if prev is None:
                    print("Starting new run")
                    logger = Logger()
                    network = torch.jit.script(self.initial_network())
                else:
                    print(f"Continuing run, first gen {gi}")
                    logger = Logger.load(self.log_path)
                    network = torch.jit.load(prev.network_path_pt)
                network.to(DEVICE)
                return gen, buffer, logger, network

    def evaluate_network(self, buffer: 'LoopBuffer', logger: Logger, network):
        setups = [
            (
                "test-buffer",
                buffer.sampler(self.train_batch_size, self.muzero_steps, self.include_final, only_last=False, test=True)
            ),
            (
                "test-last",
                buffer.sampler(self.train_batch_size, self.muzero_steps, self.include_final, only_last=True, test=True)
            ),
        ]

        network.eval()
        for prefix, sampler in setups:
            batch = sampler.next_batch_either()
            self.train_settings.evaluate_either_batch(batch, network, logger, prefix)
            sampler.close()

    def save_tmp_onnx_network(self, network, name: str) -> str:
        curr_folder = os.path.join(self.tmp_path, "curr_network")
        clean_folder(curr_folder)

        if self.muzero:
            path = os.path.join(curr_folder, f"{name}_")
            save_muzero_onnx(self.fixed_settings.game, path, network, None)
        else:
            path = os.path.join(curr_folder, f"{name}.onnx")
            save_onnx(self.fixed_settings.game, path, network, None)

        return path


@dataclass
class Generation:
    settings: 'LoopSettings'
    gi: int
    simulations_path: str
    train_path: str
    network_path_pt: str
    finished_path: str
    settings_path: str

    @classmethod
    def from_gi(cls, settings: 'LoopSettings', gi: int):
        simulations_path = os.path.join(settings.selfplay_path, f"games_{gi}")
        train_path = os.path.join(settings.training_path, f"gen_{gi}")

        return Generation(
            settings=settings,
            gi=gi,
            simulations_path=simulations_path,
            train_path=train_path,
            network_path_pt=os.path.join(train_path, "network.pt"),
            finished_path=os.path.join(train_path, "finished.txt"),
            settings_path=os.path.join(train_path, "settings.json"),
        )

    @property
    def prev(self):
        if self.gi == 0:
            return None
        return Generation.from_gi(self.settings, self.gi - 1)


class LoopBuffer:
    def __init__(self, game: Game, target_positions: int, test_fraction: float):
        self.game = game
        self.target_positions = target_positions
        self.test_fraction = test_fraction

        self.position_count = 0
        self.simulation_count = 0
        self.files: List[DataFile] = []

    def append(self, logger: Optional[Logger], file: DataFile):
        assert file.info.game == self.game, f"Expected game {self.game.name}, got game {file.info.game.name}"

        self.files.append(file)
        self.position_count += len(file.positions)
        self.simulation_count += file.info.simulation_count

        while self.position_count - len(self.files[0].positions) > self.target_positions:
            old_file = self.files[0]
            del self.files[0]

            self.position_count -= len(old_file.positions)
            self.simulation_count -= old_file.info.simulation_count
            old_file.close()

        if logger:
            logger.log("buffer", "gens", len(self.files))
            logger.log("buffer", "games", self.simulation_count)
            logger.log("buffer", "positions", self.position_count)

            info = file.info

            logger.log("gen-size", "games", info.simulation_count)
            logger.log("gen-size", "positions", info.position_count)
            logger.log("gen-game-len", "game length min", info.min_simulation_length)
            logger.log("gen-game-len", "game length mean", info.mean_simulation_length)
            logger.log("gen-game-len", "game length max", info.max_simulation_length)

            if info.root_wdl is not None:
                logger.log("gen-root-wdl", "w", info.root_wdl[0])
                logger.log("gen-root-wdl", "d", info.root_wdl[1])
                logger.log("gen-root-wdl", "l", info.root_wdl[2])

    def sampler(self, batch_size: int, unroll_steps: Optional[int], include_final: bool, only_last: bool, test: bool):
        files = [self.files[-1]] if only_last else self.files

        if test:
            range_min = 1 - self.test_fraction
            range_max = 1.0
        else:
            range_min = 0.0
            range_max = 1 - self.test_fraction

        group = DataGroup.from_files(self.game, files, range_min, range_max)

        return PositionSampler(
            group,
            batch_size,
            unroll_steps=unroll_steps,
            include_final=include_final,
            threads=1,
            include_final_for_each=False,
        )
