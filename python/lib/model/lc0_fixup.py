from typing import List

import torch
from torch import nn
from torch.nn import Conv2d

from lib.games import Game


# attempt to replicate the model LC0 uses exactly:
# https://github.com/LeelaChessZero/lczero-training/blob/3a6ed5e8cb140817cb0bd285f0f28b96988ef9e1/tf/tfprocess.py#L1185

# TODO try all of this without padding_mode="zero"
from lib.model.fixup.cifar import FixupResNet, FixupBasicBlock


class LC0FixupModel(nn.Module):
    def __init__(self, game: Game, wdl: bool, channels: int):
        super().__init__()
        assert game.name == "chess"

        self.tower = FixupResNet(game.input_channels, channels, FixupBasicBlock, [2, 2, 2])

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*8*8, 3 if wdl else 1),
        )

        # TODO extract the actually useful policy values here, so we need to transfer less data back
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, game.policy_channels, kernel_size=(1, 1)),
        )

        for mod in self.policy_head:
            nn.init.zeros_(mod.weight)

    def forward(self, input):
        common = self.tower(input)
        value_logit = self.value_head(common)
        policy_logit = self.policy_head(common)
        return value_logit, policy_logit
