import itertools
import os
from dataclasses import dataclass
from os import path
from typing import Optional, Callable

import torch
from torch import nn
from torch.optim import Adam

from games import find_game
from selfplay_client import SelfplaySettings, SelfplayClient, StartupSettings, FixedSelfplaySettings
from train import TrainSettings, train_model, TrainState, save_onnx
from util import DATA_WIDTH, GameData, load_data, DEVICE


@dataclass
class LoopSettings:
    root_path: str
    initial_network: Callable[[], nn.Module]

    buffer_gen_count: int
    test_fraction: float

    fixed_settings: FixedSelfplaySettings
    selfplay_settings: SelfplaySettings
    train_settings: TrainSettings
    train_weight_decay: float

    def new_buffer(self):
        return Buffer(self.buffer_gen_count, self.test_fraction, self.train_settings.batch_size)

    @property
    def game(self):
        return find_game(self.fixed_settings.game)


@dataclass
class Generation:
    settings: LoopSettings

    gi: int
    prev_gen_folder: str
    gen_folder: str
    games_path: str
    prev_network_path: str
    prev_network_path_onnx: str
    next_network_path: str
    next_network_path_onnx: str

    @classmethod
    def from_gi(cls, gi: int, settings: LoopSettings):
        gen_train_folder = path.join(settings.root_path, "training", f"gen_{gi}")
        prev_gen_train_folder = path.join(settings.root_path, "training", f"gen_{gi - 1}") \
            if gi != 0 else None

        return Generation(
            settings=settings,
            gi=gi,
            prev_gen_folder=prev_gen_train_folder,
            gen_folder=gen_train_folder,
            games_path=path.join(settings.root_path, "selfplay", f"games_{gi}.bin"),
            prev_network_path=path.join(prev_gen_train_folder, f"model_{settings.train_settings.epochs}_epochs.pt")
            if gi != 0 else None,
            prev_network_path_onnx=path.join(prev_gen_train_folder,
                                             f"model_{settings.train_settings.epochs}_epochs.onnx")
            if gi != 0 else None,
            next_network_path=path.join(gen_train_folder, f"model_{settings.train_settings.epochs}_epochs.pt"),
            next_network_path_onnx=path.join(gen_train_folder, f"model_{settings.train_settings.epochs}_epochs.onnx"),
        )


# TODO buffer may not fit in memory, load files in sequence instead
#   extra advantage: the last gen will always be trained on last
class Buffer:
    def __init__(self, max_gen_count: int, test_fraction: float, min_test_size: int):
        self.max_gen_count = max_gen_count
        self.test_fraction = test_fraction
        self.min_test_size = min_test_size

        self.buffer = torch.zeros(0, DATA_WIDTH)
        self.gen_lengths = []

        self.train_data: Optional[GameData] = None
        self.test_data: Optional[GameData] = None

    def push_load_path(self, games_path: str):
        train_data, test_data = load_data(games_path, self.test_fraction, limit=None)

        if len(self.gen_lengths) == self.max_gen_count:
            start = self.gen_lengths[0]
            del self.gen_lengths[0]
        else:
            start = 0

        self.gen_lengths.append(len(train_data))
        self.buffer = torch.cat([self.buffer[start:], train_data.full], dim=0)

        self.train_data = GameData(self.buffer).to(DEVICE)
        self.test_data = test_data.to(DEVICE)

    def __len__(self):
        return len(self.buffer)


def train_new_network(model, buffer: Buffer, gen: Generation):
    # model = torch.jit.load(gen.prev_network_path, map_location=DEVICE)
    state = TrainState(
        settings=gen.settings.train_settings,
        output_path=gen.gen_folder,
        train_data=buffer.train_data,
        test_data=buffer.test_data,
        optimizer=Adam(model.parameters(), weight_decay=gen.settings.train_weight_decay),
        scheduler=None
    )

    train_model(model, state)


def load_start_state(settings: LoopSettings) -> (Generation, Buffer):
    """
    Figure out how many gnerations are already finished.
    Returns the next generation to start and the buffer state up to that point.
    """
    buffer = settings.new_buffer()

    for gi in itertools.count():
        print(f"Trying generation {gi}")
        gen = Generation.from_gi(gi, settings)

        if not path.exists(gen.next_network_path):
            return gen, buffer

        buffer.push_load_path(gen.games_path)


def run_loop(settings: LoopSettings):
    game = settings.game

    print(f"Starting loop with cwd {os.getcwd()}")
    assert path.exists("./rust") and path.exists("./python"), "should be run in root STTTZero folder"

    root_path = settings.root_path
    os.makedirs(root_path, exist_ok=True)
    selfplay_folder = path.abspath(path.join(root_path, "selfplay"))

    print("Figuring out which generation to start from")
    start_gen, buffer = load_start_state(settings)

    if start_gen.gi == 0:
        print("Starting new run")

        network = torch.jit.script(settings.initial_network())
        network.to(DEVICE)

        selfplay_start_network_path = path.abspath(path.join(settings.root_path, "initial_network.onnx"))
        save_onnx(game, network, selfplay_start_network_path)
    else:
        print(f"Resuming run from gen {start_gen.gi}")

        network = torch.jit.load(start_gen.prev_network_path)
        network.to(DEVICE)

        selfplay_start_network_path = start_gen.prev_network_path_onnx

    # todo start selfplay client here at some point
    startup_settings = StartupSettings(
        output_folder=selfplay_folder,
        first_gen=start_gen.gi,
        **settings.fixed_settings.as_dict()
    )

    client = SelfplayClient()
    client.send_startup_settings(startup_settings)
    client.send_new_settings(settings.selfplay_settings)
    client.send_new_network(selfplay_start_network_path)

    for gi in itertools.count(start_gen.gi):
        print(f"Waiting for gen {gi} games")
        actual_gi = client.wait_for_file()
        assert gi == actual_gi, f"Unexpected finished generation, expected {gi} got {actual_gi}"

        gen = Generation.from_gi(gi, settings)
        os.makedirs(gen.gen_folder, exist_ok=True)

        buffer.push_load_path(gen.games_path)
        print(f"Buffer size: {len(buffer)}")

        train_new_network(network, buffer, gen)
        client.send_new_network(path.abspath(gen.next_network_path_onnx))
