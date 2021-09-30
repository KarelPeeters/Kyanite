import glob
import random
import time
from multiprocessing.pool import ThreadPool

from lib.data.file import DataFile
from lib.data.position import PositionBatch
from lib.games import Game


def main():
    pattern = "../../data/pgn-games/ccrl/*/*.json"

    game = Game.find("chess")

    # expected_size = 16 * 4 + (13 * 8 * 8 + 7) // 8 + 8 * 4 + 20 * (4 + 4)
    # print(expected_size)

    files = [DataFile(game, path) for path in glob.glob(pattern)]
    # print(len(files))

    # TODO file loading is non-thread safe!
    pool = ThreadPool(1)

    while True:
        start = time.perf_counter()

        def func(_):
            file = files[random.randrange(len(files))]
            return file[random.randrange(len(file))]

        positions = pool.map(func, range(1024))
        batch = PositionBatch(positions)

        print(f"getting batch took {time.perf_counter() - start}s")


if __name__ == '__main__':
    main()
