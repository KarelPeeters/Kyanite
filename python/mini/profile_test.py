import timeit

import torch
from torch import nn
from torch.profiler import tensorboard_trace_handler

device = "cuda"

print("Building model")


class DerpResBlock(nn.Module):
    def __init__(self, channels: int, res: bool):
        super().__init__()

        self.res = res

        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            # nn.BatchNorm2d(channels),
            # nn.ReLU(),
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            # nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        y = self.seq(x)
        if self.res:
            y.relu_()
        return y


class DerpModel(nn.Module):
    def __init__(self, channels: int, block_count: int, value_channels: int, value_size: int, res: bool):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv2d(5, channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *(DerpResBlock(channels, res) for _ in range(block_count))
        )

    def forward(self, input):
        return self.common(input)


model = DerpModel(64, 5, 8, 64, True)

model.to(device)
model.eval()

print("Generating data")
inputs = [
    torch.randn(100, 5, 9, 9, device=device)
]


def call_model():
    model(*inputs)[0].sum().item()


def without_profiler():
    for _ in range(10):
        call_model()


def with_profiler():
    with torch.profiler.profile(
            on_trace_ready=tensorboard_trace_handler("derp"),
            with_stack=True,
            with_flops=True,
    ) as profiler:
        call_model()
        profiler.step()


print("Without profiler:")
print(timeit.timeit(without_profiler, number=1))
print("With profiler:")
print(timeit.timeit(with_profiler, number=1))
