from typing import Optional, Callable

from torch import nn


class ResBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            bottleneck_channels: int,
            res: bool,
            separable: bool,
            squeeze_size: Optional[int],
    ):
        super().__init__()

        self.res = res
        self.channels = channels

        def conv(from_channels: int, to_channels: int):
            if separable:
                return [
                    # TODO what channels to use where here?
                    nn.Conv2d(from_channels, from_channels, (3, 3), padding=(1, 1), bias=False, groups=from_channels),
                    nn.Conv2d(from_channels, to_channels, (1, 1), bias=False),
                ]
            else:
                return [
                    nn.Conv2d(from_channels, to_channels, (3, 3), padding=(1, 1), bias=False)
                ]

        self.convs = nn.Sequential(
            *conv(channels, bottleneck_channels),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            *conv(bottleneck_channels, channels),
            nn.BatchNorm2d(channels),
        )

        if squeeze_size is None:
            self.squeeze = None
        else:
            self.squeeze = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, squeeze_size),
                nn.ReLU(),
                nn.Linear(squeeze_size, channels),
                nn.Sigmoid(),
            )

    def forward(self, x):
        y = self.convs(x)

        if self.squeeze is not None:
            excitation = self.squeeze(y)
            y = y * excitation[:, :, None, None]

        if self.res:
            y = y + x

        y = y.relu()
        return y


class MobileV2Block(nn.Module):
    def __init__(self, channels: int, k: int, res: bool):
        super().__init__()

        self.res = res

        self.seq = nn.Sequential(
            nn.Conv2d(channels, k * channels, (1, 1)),
            nn.BatchNorm2d(k * channels),
            nn.ReLU(),

            nn.Conv2d(k * channels, k * channels, (3, 3), padding=(1, 1), groups=k * channels),
            nn.BatchNorm2d(k * channels),
            nn.ReLU(),

            nn.Conv2d(k * channels, channels, (1, 1)),
            nn.BatchNorm2d(channels),
        )

    def forward(self, input):
        output = self.seq(input)
        if self.res:
            output = output + input
        return output


class TowerModel(nn.Module):
    def __init__(
            self,
            tower_channels: int,
            tower_depth: int,
            wdl_size: int,

            initial_act: bool,
            value_conv: bool,
            policy_conv: bool,

            block: Callable[[], nn.Module],
    ):
        super().__init__()

        self.tower = nn.Sequential(
            nn.Conv2d(3, tower_channels, (3, 3), padding=(1, 1), bias=False),
            *[
                nn.BatchNorm2d(tower_channels),
                nn.ReLU()
            ] if initial_act else [],
            *(block() for _ in range(tower_depth))
        )

        if policy_conv:
            self.policy_head = nn.Sequential(
                nn.Conv2d(tower_channels, 17, (1, 1)),
            )
        else:
            self.policy_head = nn.Sequential(
                nn.Conv2d(tower_channels, 8, (1, 1)),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * 7 * 7, 17 * 7 * 7),
                nn.Unflatten(1, (17, 7, 7))
            )

        if value_conv:
            self.wdl_head = nn.Sequential(
                nn.Conv2d(tower_channels, 1, (1, 1)),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(7*7, wdl_size),
                nn.ReLU(),
                nn.Linear(wdl_size, 3),
            )
        else:
            self.wdl_head = nn.Sequential(
                nn.AvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(tower_channels, wdl_size),
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

        common = self.tower(input)
        wdl = self.wdl_head(common)
        policy = self.policy_head(common)

        return wdl, policy
