import numpy as np
from matplotlib import pyplot as plt

w_gen = 100
l_gens = [0, 5, 10, 50, 80, 110, 150, 200, 240]
total_wdl = np.array([
    [10, 0, 0],
    [10, 0, 0],
    [10, 0, 0],
    [9, 0, 1],
    [10, 0, 0],
    [10, 0, 0],
    [10, 0, 0],
    [10, 0, 0],
    [6, 0, 4],
    [5, 0, 5],
    [6, 0, 4],
    [2, 0, 8],
    [5, 0, 5],
    [1, 0, 9],
    [4, 0, 6],
    [2, 0, 8],
    [2, 0, 8],
    [0, 0, 10]
])

total_wdl  = (total_wdl[::2] + total_wdl[1::2]) / 20
w = total_wdl[:, 0]

print(w)
plt.plot(l_gens, w, label="winrate")
plt.vlines(w_gen, 0, 1, label="baseline", color="k")
plt.title("Baseline playing against networks of different generations")
plt.xlabel("Generation")
plt.ylabel("Winrate")
plt.legend()
plt.show()