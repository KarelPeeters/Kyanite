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


# TODO think about not allowing the policy computation to access the mask?
class GoogleModel(nn.Module):
    def __init__(
            self,
            channels: int,
            blocks: int,
            wdl_channels: int, wdl_size: int,
            policy_channels: int,
            res: bool,
            squeeze_size: Optional[int], squeeze_bias: bool,
    ):
        """
        Parameters used in AlphaZero:
        channels=256
        blocks=19 or 39
        wdl_channels=1
        wdl_size=256
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

        self.wdl_head = nn.Sequential(
            nn.Conv2d(channels, wdl_channels, (1, 1), bias=False),
            nn.BatchNorm2d(wdl_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(wdl_channels * 9 * 9, wdl_size),
            nn.ReLU(),
            nn.Linear(wdl_size, 3),
        )

    def forward(self, input):
        """
        Returns `(wdl, policy)`
         * `input` is a tensor of shape (B, 5, 9, 9)
         * `wdl` is a tensor of shape (B, 3) with win/draw/loss logits
         * `policy` is a tensor of shape (B, 9, 9)
        """

        common = self.common_tower(input)
        wdl = self.wdl_head(common)
        policy = self.policy_head(common).view(-1, 9, 9)

        return wdl, policy
