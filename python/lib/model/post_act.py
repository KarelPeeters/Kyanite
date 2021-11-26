import torch
from torch import nn

from lib.games import Game


class PostActValueHead(nn.Module):
    def __init__(self, game: Game, channels: int, value_channels: int, value_hidden: int):
        super().__init__()
        self.seq = nn.Sequential(
            conv2d(channels, value_channels, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_channels * game.board_size * game.board_size, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 4)
        )

    def forward(self, common):
        return self.seq(common)


class PostActConvPolicyHead(nn.Module):
    def __init__(self, game: Game, channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            conv2d(channels, channels, 1),
            nn.ReLU(),
            conv2d(channels, game.policy_channels, 1),
        )

    def forward(self, common):
        return self.seq(common)


class PostActAttentionPolicyHead(nn.Module):
    def __init__(self, game: Game, channels: int, query_channels: int):
        super().__init__()
        assert game.name == "chess", "Attention policy only works for chess"

        self.query_channels = query_channels
        self.conv_bulk = conv2d(channels, 2 * query_channels, 1)
        self.conv_under = conv2d(channels, 3 * query_channels, 1)

    def forward(self, common):
        bulk = self.conv_bulk(common)
        under = self.conv_under(common[:, :, 7, None, :])

        q_from = bulk[:, :self.query_channels, :, :].flatten(2)
        q_to = torch.cat([
            bulk[:, self.query_channels:, :, :].flatten(2),
            under.view(-1, self.query_channels, 3 * 8)
        ], dim=2)

        policy = torch.bmm(q_from.transpose(1, 2), q_to).flatten(1)

        # TODO use new policy indexing scheme

        picked_policy = torch.gather(policy, dim=1, )

        # indexed [b][from][to_extended] before final flattening
        return policy


class PostActNetwork(nn.Module):
    def __init__(self, game: Game, depth: int, channels: int, value_head: nn.Module, policy_head: nn.Module):
        super().__init__()

        self.tower = nn.Sequential(
            conv2d(game.full_input_channels, channels, 3),
            *[Block(channels) for _ in range(depth)],
            nn.BatchNorm2d(channels),
        )

        self.value_head = value_head
        self.policy_head = policy_head

    def forward(self, input):
        common = self.tower(input)

        value_wdl = self.value_head(common)
        policy = self.policy_head(common)

        value = value_wdl[:, 0]
        wdl = value_wdl[:, 1:4]

        return value, wdl, policy


class Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            conv2d(channels, channels, 3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            conv2d(channels, channels, 3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, input):
        return input + self.seq(input)


def conv2d(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv2d:
    assert kernel_size % 2 == 1
    padding = kernel_size // 2

    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding)
    )
