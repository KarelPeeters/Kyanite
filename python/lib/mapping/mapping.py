import os
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
