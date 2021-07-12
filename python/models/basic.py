import torch
from torch import nn


class BasicModel(nn.Module):
    def __init__(self, depth: int, size: int):
        super().__init__()

        layers = [nn.Flatten()]
        prev_size = 3 * 81 + 2 * 9

        for i in range(depth):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            prev_size = size

        layers.append(nn.Linear(prev_size, 3 + 81))

        self.seq = nn.Sequential(*layers)

    def forward(self, input_mask, input_board):
        input = torch.cat([input_mask.flatten(start_dim=1), input_board.flatten(start_dim=1)], dim=1)

        output = self.seq(input)

        wdl = output[:, :3]
        policy = output[:, 3:]

        return wdl, policy
