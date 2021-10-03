import glob
import random
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
import torchvision.utils
from torchvision.utils import save_image

from lib.data.buffer import FileBuffer
from lib.data.file import DataFile
from lib.data.position import PositionBatch
from lib.games import Game


def main():
    pattern = "../../data/pgn-games/ccrl/*/*.json"

    game = Game.find("chess")

    # expected_size = 16 * 4 + (13 * 8 * 8 + 7) // 8 + 8 * 4 + 20 * (4 + 4)
    # print(expected_size)

    pool = ThreadPool(2)
    files = [DataFile(game, path) for path in glob.glob(pattern)]

    buffer = FileBuffer(game, files, pool)
    print(f"File count: {len(files)}")
    print(f"Position count: {len(buffer)}")

    while True:
        start = time.perf_counter()

        batch = buffer[np.random.randint(len(buffer), size=1024)]

        if True:
            torchvision.utils.save_image(
                batch.input_full.view(-1, 1, 8, 8), "input.png",
                nrow=batch.input_full.shape[1],
                padding=1, pad_value=0.5,
            )
            torchvision.utils.save_image(
                batch.all_wdls.view(-1, 1, 3, 3), "wdl.png",
                nrow=1,
                padding=1, pad_value=0.5,
            )
            torchvision.utils.save_image(
                batch.policy_indices.view(-1, 1, 1, batch.max_available_moves).float(), "policy_indices.png",
                nrow=1,
                padding=1, pad_value=0.5,
            )
            torchvision.utils.save_image(
                batch.policy_values.view(-1, 1, 1, batch.max_available_moves), "policy_values.png",
                nrow=1,
                padding=1, pad_value=0.5,
            )

            return

        print(f"getting batch took {time.perf_counter() - start}s")


if __name__ == '__main__':
    main()
