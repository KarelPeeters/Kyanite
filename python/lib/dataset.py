import gzip
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from lib.games import Game


class GameDataFile:
    def __init__(self, game: Game, path: str):
        f = load_as_h5_file(game, path)
        actual_game = f["game"][()].decode()
        assert game.name == actual_game, f"Expected game {game.name}, got {actual_game}"
        self.game = game

        self.positions = f["positions"]
        assert len(self.positions.shape) == 2
        assert self.positions.shape[1] == game.data_width

        ids = self.positions[:, 0:2].astype(int)
        self.game_ids = ids[:, 0]
        self.position_ids = ids[:, 1]

        self.position_count = len(self.positions)
        self.game_count = int(np.max(self.game_ids) + 1)

        # TODO this assumes all positions are in the data file, which will not be true with full_search_prob != 1
        self.game_lengths = (self.position_ids[np.diff(self.position_ids, append=0) < 0] + 1).astype(int)

    def full_dataset(self) -> 'GameDataset':
        indices = np.arange(self.position_count)
        return GameDataset(self, indices)

    def split_dataset(self, test_fraction: float) -> ('GameDataset', 'GameDataset'):
        assert 0.0 <= test_fraction <= 1
        test_count = int(test_fraction * self.game_count)

        test_game_indices = np.random.choice(self.game_count, test_count, replace=False)
        is_test_game = np.zeros(self.game_count, dtype=bool)
        is_test_game[test_game_indices] = True

        test_indices = np.argwhere(is_test_game[self.game_ids]).squeeze(1)
        train_indices = np.argwhere(~is_test_game[self.game_ids]).squeeze(1)

        return GameDataset(self, train_indices), GameDataset(self, test_indices)


class GameDataset(Dataset):
    def __init__(self, file: GameDataFile, indices: np.array):
        self.file = file
        self.indices = indices

    def __getitem__(self, index):
        return self.file.positions[self.indices[index], :]

    def __len__(self):
        return len(self.indices)


def load_as_h5_file(game: Game, path: str):
    assert path.endswith(".bin.gz") or path.endswith(".hdf5"), f"Expected .hdf5 or .bin.gz file, got {path}"
    assert os.path.exists(path), f"Path {os.path.abspath(path)} does not exist"

    if path.endswith(".bin.gz"):
        h5_path = map_bin_gz_to_hdf5(game, path)
    else:
        h5_path = path

    print(f"Loading {h5_path}")
    return h5py.File(h5_path, "r")


def map_bin_gz_to_hdf5(game: Game, bin_path: str) -> str:
    assert bin_path.endswith(".bin.gz")
    h5_path = bin_path[:-7] + ".hdf5"
    temp_h5_path = bin_path[:-7] + ".hdf5.tmp"

    # reuse the old mapping if it's up to date
    assert os.path.exists(bin_path)
    if os.path.exists(h5_path) and os.path.getmtime(h5_path) > os.path.getmtime(bin_path):
        print(f"Reusing existing {h5_path}")
        return h5_path

    print(f"Mapping {bin_path} to {h5_path}")

    # delete leftover incomplete and outdated files
    if os.path.exists(h5_path):
        os.remove(h5_path)
    if os.path.exists(temp_h5_path):
        os.remove(temp_h5_path)

    data_bytes = gzip.open(bin_path, "rb").read()
    data_np = np.frombuffer(data_bytes, dtype=np.float32).reshape(-1, game.data_width)

    with h5py.File(temp_h5_path, "w") as f:
        f.create_dataset("game", data=game.name)
        f.create_dataset(
            "positions",
            data=data_np, dtype=np.float32,
            compression="gzip", compression_opts=4, chunks=(1, game.data_width),
        )
        f.flush()

    os.rename(temp_h5_path, h5_path)
    return h5_path
