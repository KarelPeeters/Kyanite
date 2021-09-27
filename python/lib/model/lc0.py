import torch
from torch import nn
from torch.nn import Conv2d

from lib.games import Game


# attempt to replicate the model LC0 uses exactly:
# https://github.com/LeelaChessZero/lczero-training/blob/3a6ed5e8cb140817cb0bd285f0f28b96988ef9e1/tf/tfprocess.py#L1185

# TODO try all of this without padding_mode="zero"


class LC0Model(nn.Module):
    def __init__(self, game: Game, channels: int, block_count: int, wdl: bool):
        super().__init__()
        assert game.name == "chess"

        # TODO expand at least byte->float and maybe even bit->float here so we need to transfer less data
        self.tower = nn.Sequential(
            conv_block(game.input_channels_history, channels, 3, bn_scale=True),
            *(LC0Block(channels) for _ in range(block_count)),
        )

        self.value_head = nn.Sequential(
            conv_block(channels, 32, 1, bn_scale=False),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 3 if wdl else 1),
        )

        # TODO extract the actually useful policy values here, so we need to transfer less data back
        self.policy_head = nn.Sequential(
            conv_block(channels, channels, 3, bn_scale=False),
            Conv2d(channels, game.policy_channels, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, input):
        common = self.tower(input)
        value_logit = self.value_head(common)
        policy_logit = self.policy_head(common)
        return value_logit, policy_logit


class LC0Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            batch_norm(channels, scale=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            batch_norm(channels, scale=True, gamma=0),
            # TODO add squeeze-excitation here
        )

    def forward(self, x):
        return torch.relu(self.seq(x) + x)


def conv_block(in_channels: int, out_channels: int, filter_size: int, bn_scale: bool):
    assert filter_size % 2 == 1
    padding = (filter_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(filter_size, filter_size),
            padding=(padding, padding),
            bias=False,
        ),
        batch_norm(out_channels, bn_scale)
    )


def batch_norm(channels: int, gamma: float = 1.0) -> nn.Module:
    result = nn.BatchNorm2d(channels)
    nn.init.constant_(result.weight, gamma)
    return result
