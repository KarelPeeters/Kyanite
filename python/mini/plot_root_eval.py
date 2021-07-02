import os

import numpy as np
from matplotlib import pyplot

folder = "../../data/loop_scratch"

root_eval = []
game_length = []

for gen in range(28):
    path_csv = os.path.join(folder, f"gen_{gen}", "games_from_prev.csv")
    if not os.path.exists(path_csv):
        continue

    print(path_csv)
    data = np.loadtxt(path_csv, delimiter=",", ndmin=2, dtype=np.float32)
    first_rows = np.where((data[:, 4 + 81:4 + 81 + 81] == 1).all(axis=1))[0]

    root_eval.append(data[first_rows, 3].mean())
    game_length.append(len(data) / len(first_rows))

pyplot.plot(root_eval, label="root eval")
pyplot.legend()
pyplot.show()
pyplot.plot(game_length, label="game length")
pyplot.legend()
pyplot.show()
