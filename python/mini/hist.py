import numpy as np
from matplotlib import pyplot

data = np.loadtxt("../../../SuperTTTApi-rs/ignored/random_game_lengths_sttt.csv", dtype=int, delimiter=",")

bins = data.max() - data.min()
pyplot.hist(data, density=True, bins=bins)
pyplot.xlabel("Game length")
pyplot.ylabel("Frequency")
pyplot.show()
