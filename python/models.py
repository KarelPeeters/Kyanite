from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor


class ValuePolicyModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, mask: Tensor, x_tiles: Tensor, x_macros: Tensor) -> (Tensor, Tensor):
        pass


class ResBlock(nn.Module):
    def __init__(self, channels: int, res: bool):
        super().__init__()

        self.res = res

        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        y = self.seq(x)
        if self.res:
            y.relu_()
        return y


class GoogleModel(ValuePolicyModel):
    def __init__(self, channels: int, block_count: int, value_size: int, res: bool):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv2d(5, channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *(ResBlock(channels, res) for _ in range(block_count))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, (1, 1)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 9 * 9),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, (1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * 9, value_size),
            nn.ReLU(),
            nn.Linear(value_size, 1),
            nn.Tanh(),
        )

    def forward(self, mask, x_tiles, x_macros):
        device = mask.device

        range = torch.arange(9, device=device)
        os = range.view(3, 3).repeat(3, 3)
        om = range.view(3, 3).repeat_interleave(3, 0).repeat_interleave(3, 1)
        o = (9 * om + os).view(81)

        x_macros_expanded = x_macros.repeat_interleave(3, 0).repeat_interleave(3, 1)
        x_tiles_xy = x_tiles.view(-1, 2, 81)[:, :, o].view(-1, 2, 9, 9)

        input = torch.cat([
            mask.view(-1, 1, 9, 9),
            x_tiles_xy,
            x_macros_expanded.view(-1, 2, 9, 9),
        ], dim=1)

        common = self.common(input)
        value = self.value_head(common).squeeze(dim=1)
        policy = self.policy_head(common)

        # value is in range -1..1, policy are the logits
        return value, policy


class TrivialModel(ValuePolicyModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(81 + 2 * 81 + 2 * 9, 1)

    def forward(self, mask, x_tiles, x_macros):
        input = torch.cat([
            mask.view(-1, 81),
            x_tiles.view(-1, 2 * 81),
            x_macros.view(-1, 2 * 9),
        ], dim=1)

        value = torch.tanh(self.linear(input).squeeze(dim=1))
        policy = torch.ones(mask.shape[0], 81, device=mask.device)
        return value, policy
