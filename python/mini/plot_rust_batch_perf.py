import numpy as np
from matplotlib import pyplot

data = np.array([
    [1, 355.05447, 279.83554],
    [2, 761.1272, 951.8128],
    [4, 1170.9202, 1917.9475],
    [10, 1924.4404, 4379.958],
    [50, 3113.4998, 18329.854],
    [100, 2007.585, 24142.547],
    [1000, 2170.0955, 33878.113],
    [3000, 2215.8516, 31768.12],
    [5000, 2250.434, 30046.244],
])

print(data)

pyplot.loglog(data[:, 0], data[:, 1], label="cpu")
pyplot.loglog(data[:, 0], data[:, 2], label="gpu")
pyplot.ylabel("Throughput (positions/s)")
pyplot.xlabel("Batch size")
pyplot.legend()

pyplot.savefig("../ignored/cpu-vs-gpu.png")
pyplot.show()
