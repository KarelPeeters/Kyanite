import gzip
import os
from abc import ABC

import h5py
import numpy as np

from data.games import Game


class Dataset(ABC):
    def __getitem__(self, index):
        pass


class GameDataset(Dataset):
    def __init__(self, game: Game, path: str):
        super().__init__()

        f = h5py.File(path, "r")
        actual_game = f["game"][()].decode()
        assert game.name == actual_game, f"Expected game {game.name}, got {actual_game}"

        self.game = game
        self.positions = f["positions"]

        assert len(self.positions.shape) == 2
        assert self.positions.shape[1] == game.data_width

    def __getitem__(self, index):
        return self.positions[index, :]

    def __len__(self):
        return len(self.positions)

    @classmethod
    def convert_and_open(cls, game: Game, path: str):
        if path.endswith(".bin.gz"):
            h5_path = map_bin_gz_to_hdf5(game, path)
        else:
            assert path.endswith(".hdf5"), f"Expected .hdf5 or .bin.gz file, got {path}"
            h5_path = path

        print(f"Loading {h5_path}")
        return GameDataset(game, h5_path)


def map_bin_gz_to_hdf5(game: Game, bin_path: str) -> str:
    assert bin_path.endswith(".bin.gz")
    h5_path = bin_path[:-7] + ".hdf5"
    temp_h5_path = bin_path[:-7] + ".hdf5.tmp"

    # reuse the old mapping if it's up to date
    assert os.path.exists(bin_path)
    if os.path.exists(h5_path) and os.path.getmtime(h5_path) > os.path.getmtime(bin_path):
        return h5_path

    print(f"Mapping {bin_path} to {h5_path}")

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
