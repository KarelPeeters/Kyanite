import torch
from einops.layers.torch import Rearrange
from torch import nn


class AttentionTower(nn.Module):
    def __init__(
            self,
            board_size: int, depth: int,
            input_channels: int, att_channels: int, lin_channels: int,
            heads: int, dropout: float
    ):
        super().__init__()
        self.board_size = board_size

        self.expand = nn.Conv2d(input_channels, att_channels, (1, 1))
        self.embedding = nn.Parameter(torch.zeros(1, att_channels, board_size, board_size))

        self.encoders = nn.ModuleList(
            nn.TransformerEncoderLayer(att_channels, heads, lin_channels, dropout, batch_first=True)
            for _ in range(depth)
        )

        self.rearrange_before = Rearrange("b c h w -> b (h w) c", h=board_size, w=board_size)
        self.rearrange_after = Rearrange("b (h w) c -> b c h w", h=board_size, w=board_size)

    def forward(self, x):
        _, _, h, w = x.shape

        expanded = self.expand(x)
        embedded = expanded + self.embedding

        curr = self.rearrange_before(embedded)

        for encoder in self.encoders:
            curr = encoder(curr)

        reshaped = self.rearrange_after(curr)
        return reshaped
