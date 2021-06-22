from typing import Optional

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, res: bool, squeeze_size: Optional[int], squeeze_bias: bool):
        super().__init__()

        if squeeze_bias:
            assert squeeze_size is not None, "squeeze_bias without squeeze doesn't make sense"

        self.res = res
        self.squeeze_bias = squeeze_bias
        self.channels = channels

        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        if squeeze_size is None:
            self.squeeze = None
        else:
            self.squeeze = nn.Sequential(
                nn.AvgPool2d(9),
                nn.Flatten(),
                nn.Linear(channels, squeeze_size),
                nn.ReLU(),
                nn.Linear(squeeze_size, channels * (1 + squeeze_bias)),
            )

    def forward(self, x):
        y = self.convs(x)

        if self.squeeze is not None:
            weights = self.squeeze(y)

            factor = torch.sigmoid(weights[:, :self.channels, None, None])
            bias = weights[:, self.channels:, None, None]

            y = y * factor + bias

        if self.res:
            y = y + x

        y = y.relu()
        return y


class GoogleModel(nn.Module):
    def __init__(
            self,
            channels: int,
            blocks: int,
            value_channels: int, value_size: int,
            policy_channels: int,
            res: bool,
            squeeze_size: Optional[int], squeeze_bias: bool,
    ):
        """
        Parameters used in AlphaZero:
        channels=256
        blocks=19 or 39
        value_channels=1
        value_size=256
        policy_channels=2

        Oracle uses 32 channels for both heads.
        """

        super().__init__()

        self.common_tower = nn.Sequential(
            nn.Conv2d(5, channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *(ResBlock(channels, res, squeeze_size, squeeze_bias) for _ in range(blocks))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, (1, 1), bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_channels * 9 * 9, 9 * 9),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_channels, (1, 1), bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_channels * 9 * 9, value_size),
            nn.ReLU(),
            nn.Linear(value_size, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        """ Returns (value, policy) where value is in the range -1..1 and policy are the raw logits before masking."""

        common = self.common_tower(input)
        value = self.value_head(common)
        policy = self.policy_head(common)

        return value, policy


class TrivialModel:
    def __init__(self, include_mask: bool):
        super().__init__()
        self.linear = nn.Linear(include_mask * 81 + 2 * 81 + 2 * 9, 1)
        self.include_mask = include_mask

    def forward(self, mask, x_tiles, x_macros):
        input = []
        if self.include_mask:
            input.append(mask.view(-1, 81))
        input.append(x_tiles.view(-1, 2 * 81))
        input.append(x_macros.view(-1, 2 * 9))
        input = torch.cat(input, dim=1)

        value = torch.tanh(self.linear(input).squeeze(dim=1))
        policy = torch.ones(mask.shape[0], 81, device=mask.device)
        return value, policy
