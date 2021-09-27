import gzip
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from lib.games import Game
from lib.growable_array import GrowableArray


class GameDataFile:
    def __init__(self, game: Game, path: str):
        self.f = load_as_h5_file(game, path)
        actual_game = self.f["game"][()].decode()
        assert game.name == actual_game, f"Expected game {game.name}, got {actual_game}"
        self.game = game

        self.positions = self.f["positions"]
        self.position_count = self.f["position_count"][()]
        self.game_count = self.f["game_count"][()]

        self.game_ids = self.f["game_ids"]
        self.position_ids = self.f["position_ids"]

        # TODO this assumes all positions are in the data file, which will not be true when full_search_prob != 1
        self.game_lengths = (self.position_ids[np.diff(self.position_ids, append=0) < 0] + 1)

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

    def close(self):
        self.f.close()


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
        if os.path.getmtime(h5_path) > os.path.getmtime(bin_path):
            print(f"Reusing existing {h5_path}")
            return h5_path

    print(f"Mapping {bin_path} to {h5_path}")

    # delete leftover incomplete and outdated files
    if os.path.exists(h5_path):
        os.remove(h5_path)
    if os.path.exists(temp_h5_path):
        os.remove(temp_h5_path)

    # 128MB of data so we don't use too much ram but still do things in large batches
    max_batch_size = int(128 * 1024 * 1024 // 4 // game.data_width)
    print("Opening zip")
    input_reader = gzip.open(bin_path, "rb")

    with h5py.File(temp_h5_path, "w") as f:
        print("Creating dataset")
        f.create_dataset("game", data=game.name)
        positions = f.create_dataset(
            "positions",
            shape=(0, game.data_width),
            maxshape=(None, game.data_width),
            chunks=(1, game.data_width),
            compression="gzip", compression_opts=4,
            dtype=np.float32
        )

        # allocate enough space for the batch positions (which are floats, so *4)
        batch_buffer = bytearray(max_batch_size * game.data_width * 4)

        # keep basic stats around uncompressed for summary information at the end
        ids_buffer = GrowableArray(2)

        while True:
            prev_count = len(positions)

            batch_size_bytes = input_reader.readinto(batch_buffer)
            batch_size = batch_size_bytes // (game.data_width * 4)
            assert batch_size * (game.data_width * 4) == batch_size_bytes, "Unexpected length of file"

            batch_np = np.frombuffer(batch_buffer, dtype=np.float32, count=batch_size * game.data_width) \
                .reshape(batch_size, game.data_width)

            positions.resize(prev_count + batch_size, axis=0)
            positions[prev_count:, :] = batch_np

            ids_buffer.extend(batch_np[:, 0:2])
            curr_game_count = int(np.max(batch_np[:, 0]))
            print(f"Finished mapping {prev_count + batch_size} positions, from {curr_game_count} games")
            if batch_size_bytes < len(batch_buffer):
                break

        print("Writing other stuff")

        # extra information, compute it once now so we can load it later without having to decompress everything
        ids = ids_buffer.values.astype(int)
        position_ids = ids[:, 1].astype(int)
        game_ids = ids[:, 0].astype(int)

        f.create_dataset("game_ids", data=game_ids)
        f.create_dataset("position_ids", data=position_ids)

        f.create_dataset("position_count", data=len(positions))
        f.create_dataset("game_count", data=np.max(game_ids) + 1)

    os.rename(temp_h5_path, h5_path)
    return h5_path
