from typing import Optional

import torch.nn.functional as F
from torch import nn


class GlobalPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels

    def forward(self, g):
        pool_avg = F.adaptive_avg_pool2d(g, 1)
        # pool_max = F.adaptive_max_pool2d(g, 1)
        pool = pool_avg.view(-1, self.channels)
        return pool


class GlobalPoolBias(nn.Module):
    def __init__(self, channels_x: int, channels_g: int):
        super().__init__()

        self.channels_x = channels_x
        self.channels_g = channels_g

        self.gp = GlobalPool(channels_g)
        self.bn = nn.BatchNorm2d(channels_g)
        self.fc = nn.Linear(channels_g, channels_x)

    def forward(self, g):
        pool = self.gp(F.relu(self.bn(g)))
        bias = self.fc(pool).view(-1, self.channels_x, 1, 1)
        return bias


class SplitPoolBias(nn.Module):
    def __init__(self, channels: int, pool_channels: int):
        super().__init__()

        self.channels = channels
        self.pool_channels = pool_channels

        self.inner = GlobalPoolBias(channels - pool_channels, pool_channels)

    def forward(self, x):
        bias = self.inner(x[:, :self.pool_channels])
        left = x[:, self.pool_channels:]
        return left + bias


class KataBlock(nn.Module):
    def __init__(self, channels: int, pool_channels: Optional[int]):
        super().__init__()

        second_channels = (channels - pool_channels) if pool_channels is not None else channels

        self.seq = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            *([SplitPoolBias(channels, pool_channels)] if pool_channels is not None else []),
            nn.BatchNorm2d(second_channels),
            nn.ReLU(),
            nn.Conv2d(second_channels, channels, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        return x + self.seq(x)


# TODO try relu6 everywhere
# TODO properly implement pool blocks
class KataModel(nn.Module):
    def __init__(
            self,
            tower_channels: int,
            tower_depth: int,
            policy_channels: int,
            value_channels: int,
    ):
        super().__init__()

        self.tower = nn.Sequential(
            nn.Conv2d(3, tower_channels, (1, 1)),
            *[KataBlock(tower_channels, int(tower_channels / 4)) for _ in range(tower_depth)],
            nn.BatchNorm2d(tower_channels),
            nn.ReLU(),
        )

        self.wdl_head = nn.Sequential(
            nn.Conv2d(tower_channels, value_channels, (1, 1)),
            GlobalPool(value_channels),
            nn.Flatten(),
            nn.Linear(value_channels, 3)
        )

        # TODO add pooling
        # TODO try with just a single conv
        self.policy_head = nn.Sequential(
            nn.Conv2d(tower_channels, policy_channels, (1, 1)),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Conv2d(policy_channels, 17, (1, 1))
        )

    def forward(self, x):
        common = self.tower(x)
        wdl = self.wdl_head(common)
        policy = self.policy_head(common)

        return wdl, policy
