import json
import os
from dataclasses import dataclass
from io import StringIO

import numpy as np
import torch

from lib.util import DEVICE


def rel_path(name: str):
    return os.path.join(os.path.dirname(__file__), name)


def load_file_lines_to_tensor(name: str):
    with open(rel_path(name)) as f:
        data = [int(line) for line in f]

    return torch.tensor(data, device=DEVICE, dtype=torch.int64)


def load_file_lines_as_nested(name: str):
    data = []

    with open(rel_path(name)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append([int(x) for x in line.split(",")])

    return data


CHESS_FLAT_TO_CONV = load_file_lines_to_tensor("chess_flat_to_conv.txt")
assert CHESS_FLAT_TO_CONV.shape == (1880,), CHESS_FLAT_TO_CONV.shape

CHESS_FLAT_TO_ATT = load_file_lines_to_tensor("chess_flat_to_att.txt")
assert CHESS_FLAT_TO_CONV.shape == (1880,), CHESS_FLAT_TO_CONV.shape

CHESS_FLAT_TO_MOVE_INPUT = np.genfromtxt(
    rel_path("chess_flat_to_move_input.txt"),
    delimiter=",", dtype=np.int32
)[:, :-1]
assert CHESS_FLAT_TO_MOVE_INPUT.shape == (1880, 8), CHESS_FLAT_TO_MOVE_INPUT.shape

ATAXX_VALID_MOVES = load_file_lines_as_nested("ataxx_valid.txt")
assert len(ATAXX_VALID_MOVES) == 7
assert [len(x) for x in ATAXX_VALID_MOVES] == [5, 42, 113, 218, 357, 530, 737]


def load_ataxx_index_to_move_input(name: str):
    blocks = []

    with open(rel_path(name)) as f:
        s = f.read()
        for block in s.split("="):
            block = block.strip()
            if not block:
                break

            blocks.append(np.loadtxt(StringIO(block)))

    return blocks


ATAXX_INDEX_TO_MOVE_INPUT = load_ataxx_index_to_move_input("ataxx_index_to_move_input.txt")


@dataclass
class AtaxxSymmetryData:
    transpose: bool
    flip_x: bool
    flip_y: bool
    map_mv: np.array


def load_ataxx_symmetry():
    with open(rel_path("ataxx_symmetry.json"), "r") as f:
        all_dump = json.load(f)

    result = []
    for size_dump in all_dump:
        size_result = []
        for index_dump in size_dump:
            data = AtaxxSymmetryData(
                transpose=index_dump["transpose"],
                flip_x=index_dump["flip_x"],
                flip_y=index_dump["flip_y"],
                map_mv=np.array(index_dump["map_mv"]),
            )
            size_result.append(data)
        result.append(size_result)
    return result


_ATAXX_SYMMETRY_DATA = load_ataxx_symmetry()


def get_ataxx_symmetry_data(size: int, index: int):
    assert 2 <= size <= 8
    assert 0 <= index < 8
    return _ATAXX_SYMMETRY_DATA[size - 2][index]
