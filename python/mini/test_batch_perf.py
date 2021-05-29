import time

import torch
from matplotlib import pyplot

from models import GoogleModel
from util import load_data

data = load_data("../../data/esat/test_data.csv")
model = GoogleModel(channels=64, block_count=5, value_size=64, res=True)
model = torch.jit.script(model)

ROUNDS = 10

batch_sizes = [1, 2, 3, 4, 5, 10, 20, 50, 80, 100, 1000, 4000]
cpu_throughputs = torch.zeros(ROUNDS, len(batch_sizes))
cuda_throughputs = torch.zeros(ROUNDS, len(batch_sizes))

for device, throughputs in [("cpu", cpu_throughputs), ("cuda", cuda_throughputs)]:
    data = data.to(device)

    for batch_size_i, batch_size in enumerate(batch_sizes):
        print(f"Trying batch size {batch_size} on {device}")

        batch = data.pick_batch(torch.arange(batch_size))

        model.eval()
        model.to(device)

        for round in range(ROUNDS):
            start = time.perf_counter()

            value, _ = model(batch.mask.to(device), batch.x_tiles.to(device), batch.x_macros.to(device))
            # force synchronization
            value.sum().item()

            delta = time.perf_counter() - start

            throughputs[round, batch_size_i] = batch_size / delta


pyplot.plot(batch_sizes, cpu_throughputs.mean(dim=0), label="cpu")
pyplot.plot(batch_sizes, cuda_throughputs.mean(dim=0), label="gpu")
pyplot.legend()
pyplot.xlabel("batch size")
pyplot.ylabel("throughput eval/s")
pyplot.xscale("log")
pyplot.yscale("log")

pyplot.savefig("../ignored/throughputs.png")
pyplot.show()
