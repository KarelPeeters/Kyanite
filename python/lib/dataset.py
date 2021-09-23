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
        self.position_count = f["position_count"][()]
        self.game_count = f["game_count"][()]
        self.game_ids = f["game_ids"]

        assert len(self.positions.shape) == 2
        assert self.positions.shape[1] == game.data_width

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

    # reuse the old mapping if it's up to date (with the bin file *and* the source of this file
    assert os.path.exists(bin_path)
    if os.path.exists(h5_path):
        h5_time = os.path.getmtime(h5_path)
        if h5_time > os.path.getmtime(bin_path) and h5_time > os.path.getmtime(__file__):
            print(f"Reusing existing {h5_path}")
            return h5_path

    print(f"Mapping {bin_path} to {h5_path}")

    # delete leftover incomplete and outdated files
    if os.path.exists(h5_path):
        os.remove(h5_path)
    if os.path.exists(temp_h5_path):
        os.remove(temp_h5_path)

    data_bytes = gzip.open(bin_path, "rb").read()
    positions = np.frombuffer(data_bytes, dtype=np.float32).reshape(-1, game.data_width)

    with h5py.File(temp_h5_path, "w") as f:
        f.create_dataset("game", data=game.name)
        f.create_dataset(
            "positions",
            data=positions, dtype=np.float32,
            compression="gzip", compression_opts=4, chunks=(1, game.data_width),
        )

        # extra information, compute it once now so we can load it later without having to decompress everything
        ids = positions[:, 0:2].astype(int)

        position_ids = ids[:, 1]
        game_ids = ids[:, 0]
        f.create_dataset("game_ids", data=game_ids)
        f.create_dataset("position_ids", data=position_ids)

        f.create_dataset("position_count", data=len(positions))
        f.create_dataset("game_count", data=np.max(game_ids) + 1)

        # TODO this assumes all positions are in the data file, which will not be true when full_search_prob != 1
        game_lengths = (position_ids[np.diff(position_ids, append=0) < 0] + 1)
        f.create_dataset("game_lengths", data=game_lengths)

        f.flush()

    os.rename(temp_h5_path, h5_path)
    return h5_path
