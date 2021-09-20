from pathlib import Path
from typing import Optional

import numpy
import numpy as np
import torch

from games import Game

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")


class GameData:
    def __init__(self, game: Game, full):
        self.game = game

        assert len(full.shape) == 2
        assert full.shape[1] == game.data_width, f"Expected size {game.data_width}, got {full.shape[1]}"

        self.full = full

        i = 0

        def take(length: int):
            nonlocal i
            i += length
            return full[:, i - length:i]

        self.wdl_final = take(3)
        self.wdl_est = take(3)

        self.policy_mask = take(game.policy_size).view(-1, *game.policy_shape)
        self.policy = take(game.policy_size).view(-1, *game.policy_shape)

        self.board = take(game.input_size).view(-1, *game.input_shape)

        assert i == game.data_width

    def to(self, device):
        return GameData(self.game, self.full.to(device))

    def random_symmetry(self):
        # TODO re-implement this, generic over the game
        return self

    def __getitem__(self, indices):
        return GameData(self.game, self.full[indices, :])

    def __len__(self):
        return len(self.full)


def load_data_multiple(game: Game, paths: [str], test_fraction: float, limit_each: Optional[int] = None) -> (
        GameData, GameData):
    """
    All paths bust be .bin files
    This function does not actually shuffle train and test data itself, but games are randomly split between them.
    This is okay because they're shuffled during training anyway.
    """
    data_width = game.data_width

    load_count = -1 if limit_each is None else limit_each * (1 + data_width)
    games = torch.zeros(0, 1 + data_width)

    for path in paths:
        path = Path(path)

        assert path.suffix == ".bin", f"Unexpected extension '{path.suffix}'"

        print(f"Loading data")
        part_games = numpy.fromfile(path, dtype=np.float32, count=load_count)
        if len(part_games) == 0:
            raise ValueError(f"Empty file {path}")

        part_games = torch.tensor(part_games).view(-1, 1 + data_width)
        games = torch.cat([games, part_games], dim=0)

    print("Splitting data")
    game_ids = games[:, 0].round().long()
    game_count = game_ids.max() + 1
    full = games[:, 1:]

    perm_games = torch.randperm(game_count)
    split_index = int((1 - test_fraction) * game_count)

    train_mask = perm_games[game_ids] < split_index
    train_data = GameData(game, full[train_mask, :])
    test_data = GameData(game, full[~train_mask, :])

    print(f"Train size {len(train_data)}, test size {len(test_data)}")
    return train_data, test_data


def load_data(game: Game, path: str, test_fraction: float, limit: Optional[int]) -> (GameData, GameData):
    return load_data_multiple(game, [path], test_fraction, limit)


def o_tensor(device):
    r = torch.arange(9, device=device)
    os_full = r.view(3, 3).repeat(3, 3)
    om_full = r.view(3, 3).repeat_interleave(3, 0).repeat_interleave(3, 1)
    o_full = (9 * om_full + os_full).view(81)
    return o_full
