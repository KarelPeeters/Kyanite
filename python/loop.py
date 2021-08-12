import itertools
import os
from dataclasses import dataclass
from os import path
from typing import Optional, Callable

import torch
from torch import nn
from torch.optim import Adam

from selfplay_client import SelfplaySettings, SelfplayClient, StartupSettings
from train import TrainSettings, train_model, TrainState
from util import DATA_WIDTH, GameData, load_data, DEVICE


@dataclass
class LoopSettings:
    root_path: str
    initial_network: Callable[[], nn.Module]

    buffer_gen_count: int
    test_fraction: float

    startup_settings: StartupSettings
    selfplay_settings: SelfplaySettings
    train_settings: TrainSettings
    train_weight_decay: float

    def new_buffer(self):
        return Buffer(self.buffer_gen_count, self.test_fraction, self.train_settings.batch_size)


@dataclass
class Generation:
    settings: LoopSettings

    gi: int
    prev_gen_folder: str
    gen_folder: str
    games_path: str
    prev_network_path: str
    next_network_path: str
    next_network_path_onnx: str

    @classmethod
    def from_gi(cls, gi: int, settings: LoopSettings):
        gen_folder = path.join(settings.root_path, f"gen_{gi}")
        prev_gen_folder = path.join(settings.root_path, f"gen_{gi - 1}") \
            if gi != 0 else None

        return Generation(
            settings=settings,
            gi=gi,
            prev_gen_folder=prev_gen_folder,
            gen_folder=gen_folder,
            games_path=path.join(settings.root_path, "selfplay_games", f"games_{gi}.bin"),
            prev_network_path=path.join(prev_gen_folder, f"model_{settings.train_settings.epochs}_epochs.pt")
            if gi != 0 else settings.initial_network,
            next_network_path=path.join(gen_folder, f"model_{settings.train_settings.epochs}_epochs.pt"),
            next_network_path_onnx=path.join(gen_folder, f"model_{settings.train_settings.epochs}_epochs.onnx"),
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


def find_last_finished_gen(settings: LoopSettings) -> Optional[int]:
    for gi in itertools.count():
        gen = Generation.from_gi(gi, settings)

        if not path.exists(gen.next_network_path):
            if gi >= 1:
                return gi - 1
            return None


def load_resume_buffer(settings: LoopSettings, last_finished_gi: int) -> Buffer:
    buffer = settings.new_buffer()
    for gi in range(last_finished_gi + 1):
        gen = Generation.from_gi(gi, settings)
        buffer.push_load_path(gen.games_path)
    return buffer


def save_onnx(network, onnx_path: str):
    print(f"Saving model to {onnx_path}")
    network.eval()
    example_input = torch.zeros(1, 3, 7, 7, device=DEVICE)
    example_outputs = network(example_input)
    torch.onnx.export(
        model=network,
        args=example_input,
        f=onnx_path,
        example_outputs=example_outputs,
        input_names=["input"],
        output_names=["wdl", "policy"],
        dynamic_axes={"input": {0: "batch_size"}, "wdl": {0: "batch_size"}, "policy": {0: "batch_size"}},
    )


def run_loop(settings: LoopSettings):
    assert settings.startup_settings.output_folder == "", "Output folder is set automatically, don't set it manually"
    settings.startup_settings.output_folder = path.abspath(path.join(settings.root_path, "selfplay_games"))

    print(f"Starting loop in directory {os.getcwd()}")
    assert path.exists("./rust") and path.exists("./python"), "should be run in root STTTZero folder"

    # check if we're resuming a run and restore the buffer if so
    last_finished_gi = find_last_finished_gen(settings)
    if last_finished_gi is not None:
        start_gi = last_finished_gi + 1
        print(f"Resuming {settings.root_path}, restarting with gen {start_gi}")
        buffer = load_resume_buffer(settings, last_finished_gi)
    else:
        print("Starting new run from gen 0")
        start_gi = 0
        buffer = settings.new_buffer()

    assert start_gi == 0, "Continuing a run is not supported yet"

    network = torch.jit.script(settings.initial_network())
    network.to(DEVICE)

    initial_network_path = path.abspath(path.join(settings.root_path, "initial_network.onnx"))
    save_onnx(network, initial_network_path)

    # todo start selfplay client here at some point

    client = SelfplayClient()
    client.send_startup_settings(settings.startup_settings)
    client.send_new_settings(settings.selfplay_settings)
    client.send_new_network(initial_network_path)

    for gi in itertools.count():
        print(f"Waiting for gen {gi} games")
        actual_gi = client.wait_for_file()
        assert gi == actual_gi

        gen = Generation.from_gi(gi, settings)
        os.makedirs(gen.gen_folder, exist_ok=True)

        buffer.push_load_path(gen.games_path)
        print(f"Buffer size: {len(buffer)}")

        train_new_network(network, buffer, gen)

        save_onnx(network, gen.next_network_path_onnx)
        client.send_new_network(path.abspath(gen.next_network_path_onnx))
