import dataclasses
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Union


@dataclass
class StartupSettings:
    output_folder: str
    first_gen: int

    game: str
    threads_per_device: int
    batch_size: int
    games_per_gen: int

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass
class SelfplaySettings:
    max_game_length: int
    exploration_weight: float
    use_value: bool
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


CONNECT_TRY_PERIOD = 1.0


def connect_to_selfplay_server() -> socket.socket:
    while True:
        last_attempt_start = time.time()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", 63105))
            return s
        except ConnectionRefusedError as e:
            print(e)

        delay = (time.time() - last_attempt_start) - CONNECT_TRY_PERIOD
        if delay > 0:
            time.sleep(delay)


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
        path = os.path.abspath(path)
        self.send({"NewNetwork": path})

    def send_wait_for_new_network(self):
        self.send("WaitForNewNetwork")

    def send_stop(self):
        self.send("Stop")

    def wait_for_file(self) -> int:
        message = json.loads(self.f.readline())
        print(f"Received message {message}")
        return message["FinishedFile"]["index"]
