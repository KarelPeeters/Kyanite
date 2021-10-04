# attempt to replicate the model LC0 uses exactly:
# "current" latest version:
# https://github.com/LeelaChessZero/lczero-training/blob/3a6ed5e8cb140817cb0bd285f0f28b96988ef9e1/tf/tfprocess.py#L1185
# version at the time of the common dataset:
# https://github.com/LeelaChessZero/lczero-training/blob/77fbfba230e3f309d474fe60dd9d469a76fc28f2/tf/tfprocess.py#L397-L530
import torch
from torch import nn

from lib.games import Game


class LCZOldNetwork(nn.Module):
    def __init__(self, game: Game, channels: int, depth: int):
        super().__init__()
        assert game.name == "chess"
        self.policy_channels = game.policy_channels

        self.tower = nn.Sequential(
            conv_block(3, game.full_input_channels, channels),
            *[ResBlock(channels) for _ in range(depth)],
        )

        self.policy_head = nn.Sequential(
            conv_block(1, channels, 32),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, game.policy_channels * 8 * 8),
        )

        self.value_head = nn.Sequential(
            conv_block(1, channels, 32),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, input):
        common = self.tower(input)
        value_wdl = self.value_head(common)
        policy = self.policy_head(common).view(-1, self.policy_channels, 8, 8)

        value = value_wdl[:, 0]
        wdl = value_wdl[:, 1:4]
        return value, wdl, policy


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            batch_norm(channels, scale=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            batch_norm(channels, scale=False),
        )

    def forward(self, x):
        return (self.seq(x) + x).relu()


class GlobalBias(nn.Module):
    def forward(self, input):
        channel_mean = input.flatten(2).mean(2)
        return input + channel_mean[:, :, None, None]


class GlobalDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(8 * 8, 8 * 8),
            nn.ReLU(),
        )

    def forward(self, input):
        picked = input[:, 0, :, :].flatten(1)
        output = self.seq(picked)
        return torch.cat([
            output.view(-1, 1, 8, 8),
            input[:, 1:, :, :],
        ], dim=1)


def conv_block(kernel_size: int, in_channels: int, out_channels: int) -> nn.Module:
    assert kernel_size % 2 == 1
    padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=(padding, padding)),
        batch_norm(out_channels, scale=False),
        nn.ReLU(),
    )


def batch_norm(channels: int, scale: bool, gamma: float = 1.0) -> nn.Module:
    # TODO virtual batch size?
    if scale:
        d = nn.BatchNorm2d(channels)
        d.weight.fill_(gamma)
        return d
    else:
        return BatchNormNoScale(channels)


class BatchNormNoScale(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.bn(x) + self.bias
