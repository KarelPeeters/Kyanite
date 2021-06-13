import time

import torch
from matplotlib import pyplot

from models import GoogleModel, ValuePolicyModel
from util import load_data, o_tensor


class ModelWrapper(ValuePolicyModel):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, o, mask, tiles, macros):
        return self.inner(o, mask.float(), tiles.float(), macros.float())


def main():
    data = load_data("../../data/esat/test_data.csv")
    o = o_tensor("cpu")

    ROUNDS = 20
    DEVICES = ["cuda"]
    BATCH_SIZES = [100, 200, 500, 1000]
    PARALLEL_COUNT = 1

    google = GoogleModel(channels=64, blocks=5, value_channels=1, value_size=64, res=True)
    DTYPES = [
        (torch.int8, torch.jit.script(ModelWrapper(google))),
        (torch.float16, torch.jit.script(ModelWrapper(google))),
        (torch.float32, torch.jit.script(google)),
    ]

    throughputs = torch.zeros(len(DTYPES), len(DEVICES), len(BATCH_SIZES))

    for ti, (dtype, model) in enumerate(DTYPES):
        for di, device in enumerate(DEVICES):
            data = data.to(device)
            model.eval()
            model.to(device)
            o = o.to(device)

            for bi, batch_size in enumerate(BATCH_SIZES):
                print(f"Trying batch size {batch_size} on {device} with type {dtype}: ", end="")

                totals = [torch.tensor(0)] * PARALLEL_COUNT

                batch = data.pick_batch(torch.arange(batch_size))
                mask = batch.mask.type(dtype)
                x_tiles = batch.x_tiles.type(dtype)
                x_macros = batch.x_macros.type(dtype)

                start = time.perf_counter()

                for _ in range(ROUNDS):
                    value, _ = model(o, mask.to(device), x_tiles.to(device), x_macros.to(device))

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
