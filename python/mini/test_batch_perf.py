import time

import torch
from matplotlib import pyplot
from torch import nn

from models import GoogleModel
from util import load_data, print_param_count


class ModelWrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x.float())


def main():
    data, _ = load_data("../../data/ataxx/mcts_games.bin", 0.0, None)

    ROUNDS = 20
    DEVICES = ["cpu", "cuda"]
    BATCH_SIZES = [100, 200, 500, 1000]
    PARALLEL_COUNT = 2

    google = GoogleModel(
        channels=64, blocks=5,
        wdl_channels=1, wdl_size=64,
        res=True, separable=True,
        squeeze_size=None, squeeze_bias=False
    )
    print_param_count(google)
    DTYPES = [
        # (torch.int8, torch.jit.script(ModelWrapper(google))),
        # (torch.float16, torch.jit.script(ModelWrapper(google))),
        (torch.float32, torch.jit.script(google)),
    ]

    throughputs = torch.zeros(len(DTYPES), len(DEVICES), len(BATCH_SIZES))

    for ti, (dtype, model) in enumerate(DTYPES):
        for di, device in enumerate(DEVICES):
            data = data.to(device)
            model.eval()
            model.to(device)

            for bi, batch_size in enumerate(BATCH_SIZES):
                print(f"Trying batch size {batch_size} on {device} with type {dtype}: ", end="")

                totals = [torch.tensor(0)] * PARALLEL_COUNT

                batch = data[:batch_size]

                start = time.perf_counter()

                for _ in range(ROUNDS):
                    value, _ = model(batch.board)

                    # force synchronization to prevent out of memory
                    totals[:-1] = totals[1:]
                    totals[-1] = value.sum()
                    totals[0].item()

                delta = time.perf_counter() - start
                throughput = batch_size * ROUNDS / delta
                throughputs[ti, di, bi] = throughput

                print(throughput)

    for di, device in enumerate(DEVICES):
        for ti, (dtype, _) in enumerate(DTYPES):
            pyplot.plot(BATCH_SIZES, throughputs[ti, di, :], label=f"{device} {dtype}")

    pyplot.legend()
    pyplot.xlabel("batch size")
    pyplot.ylabel("throughput eval/s")
    pyplot.xscale("log")
    pyplot.yscale("log")

    pyplot.savefig("../ignored/throughputs.png")
    pyplot.show()


if __name__ == '__main__':
    main()
