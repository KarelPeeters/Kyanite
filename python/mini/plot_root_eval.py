import itertools
import os

import numpy as np
from matplotlib import pyplot

folder = "../../data/loop_wdl_0"

root_wdl = []
game_length = []

for gen in itertools.count():
    path_csv = os.path.join(folder, f"gen_{gen}", "games_from_prev.csv")
    if not os.path.exists(path_csv):
        print(f"Stopping at {gen}, csv file does not exist")
        break

    print(path_csv)
    data = np.loadtxt(path_csv, delimiter=",", ndmin=2, dtype=np.float32)
    first_rows = np.where((data[:, 6 + 81:6 + 81 + 81] == 1).all(axis=1))[0]

    root_wdl.append(data[first_rows, 3:6].mean(axis=0))
    game_length.append(len(data) / len(first_rows))

gen_count = len(root_wdl)
root_wdl = np.array(root_wdl).transpose()

pyplot.stackplot(np.arange(gen_count), root_wdl, labels=["win", "draw", "loss"])
pyplot.legend()
pyplot.title("Zero estimated WDL of the empty board")
pyplot.show()

pyplot.plot(game_length, label="game length")
pyplot.legend()
pyplot.title("Average game length")
pyplot.show()
