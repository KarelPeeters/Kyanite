# attempt to replicate the model LC0 uses exactly:
# "current" latest version:
# https://github.com/LeelaChessZero/lczero-training/blob/3a6ed5e8cb140817cb0bd285f0f28b96988ef9e1/tf/tfprocess.py#L1185
# version at the time of the common dataset:
# https://github.com/LeelaChessZero/lczero-training/blob/77fbfba230e3f309d474fe60dd9d469a76fc28f2/tf/tfprocess.py#L397-L530
from typing import Tuple

import torch
from torch import nn

from lib.games import Game


class LCZOldPreNetwork(nn.Module):
    def __init__(self, game: Game, depth: int, channels: int, value_channels: int, value_hidden: int):
        super().__init__()
        self.policy_channels = game.policy_channels
        self.b = game.board_size

        self.tower = nn.Sequential(
            nn.Conv2d(game.full_input_channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            *[PreResBlock(channels) for _ in range(depth)],
            batch_norm(channels, scale=True),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            conv_2d(1, channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            conv_2d(1, channels, self.policy_channels),
        )

        self.value_head = nn.Sequential(
            conv_2d(1, channels, value_channels),
            nn.BatchNorm2d(value_channels),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(value_channels * self.b * self.b, value_hidden),
            nn.BatchNorm1d(value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 4),
        )

    def forward(self, input):
        common = self.tower(input)
        value_wdl = self.value_head(common)
        policy = self.policy_head(common).view(-1, self.policy_channels, self.b, self.b)

        value = value_wdl[:, 0]
        wdl = value_wdl[:, 1:4]
        return value, wdl, policy


class PreResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            batch_norm(channels, scale=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            batch_norm(channels, scale=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        return x + self.seq(x)


def conv_2d(kernel_size: int, in_channels: int, out_channels: int) -> nn.Module:
    assert kernel_size % 2 == 1
    padding = (kernel_size - 1) // 2

    return nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=(padding, padding))


def batch_norm(channels: int, scale: bool) -> nn.Module:
    if scale:
        return nn.BatchNorm2d(channels)
    else:
        return BatchNormNoScale(channels)


class BatchNormNoScale(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.bn(x) + self.bias
