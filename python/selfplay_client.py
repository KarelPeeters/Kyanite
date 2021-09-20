import dataclasses
import json
import socket
from dataclasses import dataclass
from typing import Union

from games import Game


@dataclass
class FixedSelfplaySettings:
    game: str
    threads_per_device: int
    batch_size: int
    games_per_gen: int

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass
class StartupSettings:
    game: str
    output_folder: str
    threads_per_device: int
    batch_size: int
    games_per_gen: int
    first_gen: int

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass
class SelfplaySettings:
    max_game_length: int
    exploration_weight: float
    random_symmetries: bool
    keep_tree: bool
    temperature: float
    zero_temp_move_count: int
    dirichlet_alpha: float
    dirichlet_eps: float
    full_search_prob: float
    full_iterations: int
    part_iterations: int
    cache_size: int

    def as_dict(self):
        return dataclasses.asdict(self)


def connect_to_selfplay_server() -> socket.socket:
    while True:
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            s.connect(("::1", 63105))
            return s
        except ConnectionRefusedError as e:
            print(e)


class SelfplayClient:
    def __init__(self):
        self.s = connect_to_selfplay_server()
        self.f = self.s.makefile("r")

    def send(self, message: Union[dict, str]):
        s = json.dumps(message)
        print(f"Sending '{s}'")
        self.s.send((s + "\n").encode())

    def send_startup_settings(self, settings: StartupSettings):
        self.send({"StartupSettings": settings.as_dict()})

    def send_new_settings(self, settings: SelfplaySettings):
        self.send({"NewSettings": settings.as_dict()})

    def send_new_network(self, path: str):
        self.send({"NewNetwork": path})

    def send_stop(self):
        self.send("Stop")

    def wait_for_file(self) -> int:
        message = json.loads(self.f.readline())
        return message["FinishedFile"]["index"]
