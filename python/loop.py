import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass
from os import path
from typing import Optional

import torch
from torch.optim import Adam

from train import TrainSettings, train_model, TrainState
from util import DATA_WIDTH, GenericData, load_data, DEVICE, GoogleData


@dataclass
class SelfplaySettings:
    position_count: int

    inf_temp_move_count: int
    dirichlet_alpha: float
    dirichlet_eps: float

    iterations: int
    exploration_weight: float

    batch_size: int
    threads_per_device: int

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class LoopSettings:
    root_path: str
    initial_network: str

    generations: int
    buffer_gen_count: int
    test_fraction: float

    selfplay_settings: SelfplaySettings
    train_settings: TrainSettings
    train_weight_decay: float


@dataclass
class Generation:
    settings: LoopSettings

    gi: int
    prev_gen_folder: str
    gen_folder: str
    games_path: str
    prev_network_path: str

    @classmethod
    def from_gi(cls, gi: int, settings: LoopSettings):
        gen_folder = path.join(settings.root_path, f"gen_{gi}")
        prev_gen_folder = path.join(settings.root_path, f"gen_{gi - 1}") \
            if gi != 0 else None

        os.makedirs(gen_folder, exist_ok=True)

        return Generation(
            settings=settings,
            gi=gi,
            prev_gen_folder=prev_gen_folder,
            gen_folder=gen_folder,
            games_path=path.join(gen_folder, "games_from_prev.csv"),
            prev_network_path=path.join(prev_gen_folder, f"model_{settings.train_settings.epochs}_epochs.pt")
            if gi != 0 else settings.initial_network,
        )


class Buffer:
    def __init__(self, max_gen_count: int, test_fraction: float, min_test_size: int):
        self.max_gen_count = max_gen_count
        self.test_fraction = test_fraction
        self.min_test_size = min_test_size

        self.buffer = torch.zeros(0, DATA_WIDTH)
        self.gen_lengths = []

        self.train_data: Optional[GoogleData] = None
        self.test_data: Optional[GoogleData] = None

    def push(self, data: GenericData):
        print("Push call")
        split_index = min(int(self.test_fraction * len(data)), self.min_test_size)
        test_data = data.pick_batch(slice(None, split_index))
        train_data = data.pick_batch(slice(split_index, None))

        if len(self.gen_lengths) == self.max_gen_count:
            start = self.gen_lengths[0]
            del self.gen_lengths[0]
        else:
            start = 0

        self.gen_lengths.append(len(train_data))
        self.buffer = torch.cat([self.buffer[start:], train_data.full], dim=0)

        self.train_data = GoogleData.from_generic(GenericData(self.buffer)).to(DEVICE)
        # TODO the full set be used as test data? should we just drop the idea of test data altogether?
        self.test_data = GoogleData.from_generic(test_data).to(DEVICE)

    def __len__(self):
        return len(self.buffer)


def generate_selfplay_games(gen: Generation) -> GenericData:
    arg_dict = gen.settings.selfplay_settings.to_dict()
    arg_dict["output_path"] = gen.games_path
    arg_dict["network_path"] = gen.prev_network_path

    command = "cargo run --manifest-path rust/Cargo.toml --release --bin selfplay_cmd".split(" ") + [
        json.dumps(arg_dict)]
    print(f"Running command {command}")

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    for line in p.stdout:
        print(line, end="")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, command)

    return load_data(gen.games_path, shuffle=True)


def train_new_network(buffer: Buffer, gen: Generation):
    model = torch.jit.load(gen.prev_network_path, map_location=DEVICE)
    state = TrainState(
        settings=gen.settings.train_settings,
        output_path=gen.gen_folder,
        train_data=buffer.train_data,
        test_data=buffer.test_data,
        optimizer=Adam(model.parameters(), weight_decay=gen.settings.train_weight_decay),
        scheduler=None
    )

    train_model(model, state)


def run_loop(settings: LoopSettings):
    print(f"Starting loop in directory {os.getcwd()}")
    assert os.path.exists("./rust") and os.path.exists("./python"), "should be run in root STTTZero folder"

    buffer = Buffer(settings.buffer_gen_count, settings.test_fraction, settings.train_settings.batch_size)

    for gi in range(settings.generations):
        print(f"Starting generation {gi}")

        gen = Generation.from_gi(gi, settings)

        new_games = generate_selfplay_games(gen)
        buffer.push(new_games)

        print(f"Buffer size: {len(buffer)}")

        train_new_network(buffer, gen)
