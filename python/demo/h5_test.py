import itertools
import os
import random

from lib.games import Game
from lib.loop import Buffer
from lib.train import batch_loader


def main():
    folder = "../../data/pgn-games/cclr/test/"
    paths = [os.path.join(folder, f) for f in os.listdir((folder)) if f.endswith(".bin.gz")]

    game = Game.find("chess")
    buffer = Buffer(game, int(1e6), 0.05)


    batch_size = 1024

    for path in paths:
        buffer.append(None, path)

        loader = batch_loader(buffer.full_train_dataset(), batch_size)
        for batch in itertools.islice(loader, 32):
            print(batch.shape)


if __name__ == '__main__':
    main()
