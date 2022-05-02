from dataclasses import dataclass

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn


class AttentionTower(nn.Module):
    def __init__(
            self,
            board_size: int, input_channels: int,
            depth: int,
            d_model: int, heads: int, d_k: int, d_v: int, d_ff: int,
            dropout: float
    ):
        super().__init__()
        self.board_size = board_size

        self.expand = nn.Conv2d(input_channels, d_model, (1, 1))
        self.embedding = nn.Parameter(torch.zeros(1, d_model, board_size, board_size))

        self.encoders = nn.ModuleList(
            EncoderLayer(d_model, heads, d_k, d_v, d_ff, dropout)
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


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, d_v: int, d_ff: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.d_k = d_v
        self.d_v = d_v
        self.d_ff = d_ff

        self.dk_total = heads * d_k

        # TODO better (DeepNorm) initialization
        self.project_qkv = nn.Conv1d(d_model, heads * (2 * d_k + d_v), 1)
        self.project_out = nn.Conv1d(heads * d_v, d_model, 1)

        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            nn.ReLU(),
            nn.Conv1d(d_ff, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout)

        self.norm_att = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

    def forward_with_weights(self, input):
        qkv = run_conv(self.project_qkv, input)

        q = qkv[:, :, :self.dk_total]
        k = qkv[:, :, self.dk_total:2 * self.dk_total]
        v = qkv[:, :, 2 * self.dk_total:]

        lookup, weights = multi_head_attention(q, k, v, self.heads)
        projected = run_conv(self.project_out, lookup)

        att_result = self.norm_att(input + self.dropout(projected))
        ff_result = run_conv(self.ff, att_result)
        result = self.norm_ff(att_result + self.dropout(ff_result))

        return result, weights

    def forward(self, input):
        result, _ = self.forward_with_weights(input)
        return result


def run_conv(f, x):
    x_conv = einops.rearrange(x, "b n c -> b c n")
    y_conv = f(x_conv)
    y = einops.rearrange(y_conv, "b c n -> b n c")
    return y


def multi_head_attention(q, k, v, heads: int):
    shapes = check_att_shapes(q, k, v)
    assert shapes.dk_total % heads == 0 and shapes.dv_total % heads == 0

    # shuffle the input
    q_split = einops.rearrange(q, "b m (h k) -> (b h) m k", h=heads, m=shapes.m)
    k_split = einops.rearrange(k, "b n (h k) -> (b h) k n", h=heads, n=shapes.n)
    v_split = einops.rearrange(v, "b n (h v) -> (b h) n v", h=heads, n=shapes.n)

    # actual attention
    logits_split = torch.bmm(q_split, k_split)
    att_split = torch.softmax(logits_split, -1)
    result_split = torch.bmm(att_split, v_split)

    # shuffle the output back
    result = einops.rearrange(result_split, "(b h) m v -> b m (h v)", h=heads, m=shapes.m)
    return result, att_split


@dataclass
class AttShapes:
    b: int  # batch size
    n: int  # input sequence length (for k and v)
    m: int  # output sequence length (for q)
    dk_total: int  # key size (for q and k)
    dv_total: int  # value size


def check_att_shapes(q, k, v) -> AttShapes:
    b0, m, dk0 = q.shape
    b1, n0, dk1 = k.shape
    b2, n1, dv = v.shape

    assert b0 == b1 == b2, "Batch size mismatch"
    assert n0 == n1, "Input seq length mismatch"
    assert dk1 == dk0, "Key size mismatch"

    return AttShapes(b=b0, n=n0, m=m, dk_total=dk0, dv_total=dv)
