from typing import NamedTuple

import torch
import torch.nn.functional as nnf
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
        self.d_model = d_model

        alpha = (2 * depth) ** (1 / 4)
        beta = (8 * depth) ** (-1 / 4)

        self.expand = nn.Linear(input_channels, d_model, bias=False)
        self.embedding = nn.Parameter(torch.randn(board_size * board_size, d_model))

        self.encoders = nn.ModuleList(
            EncoderLayer(d_model, heads, d_k, d_v, d_ff, dropout, alpha=alpha, beta=beta)
            for _ in range(depth)
        )

    def forward(self, x):
        b, d_in, h, w = x.shape

        # "b c h w -> (h w) b c"
        shaped = x.permute(2, 3, 0, 1).view(h * w, b, d_in)

        expanded = self.expand(shaped.reshape(h * w * b, d_in)).view(h * w, b, self.d_model)
        curr = expanded + self.embedding.unsqueeze(1)

        for encoder in self.encoders:
            curr = encoder(curr)

        # "(h w) b c -> b c h w"
        reshaped = curr.view((h, w, b, self.d_model)).permute((2, 3, 0, 1))
        return reshaped


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int, heads: int, d_k: int, d_v: int, d_ff: int,
            dropout: float,
            alpha: float = 1.0, beta: float = 1.0
    ):
        super().__init__()

        # save model sizes
        assert d_model % heads == 0

        self.d_model = d_model
        self.d_ff = d_ff

        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v

        self.d_kqv = 2 * d_k + d_v
        self.d_k_total = heads * self.d_k

        # build inner layers
        self.project_qkv = nn.Linear(d_model, heads * self.d_kqv, bias=False)
        self.project_out = nn.Linear(heads * d_v, d_model, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_att = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(d_model, elementwise_affine=False)

        # initialize weights according to DeepNet/DeepNorm paper
        self.alpha = alpha

        project_q = self.project_qkv.weight[:self.d_k_total, :]
        project_k = self.project_qkv.weight[self.d_k_total:2 * self.d_k_total, :]
        project_v = self.project_qkv.weight[2 * self.d_k_total:, :]

        for w in [self.ff[0].weight, self.ff[2].weight, project_v, self.project_out.weight]:
            nn.init.xavier_normal_(w, gain=beta)
        for w in [project_q, project_k]:
            nn.init.xavier_normal_(w, gain=1)

    def forward_with_weights(self, input):
        # input & output: (n, b, d_model)

        heads = self.heads
        (n, b, c) = input.shape
        assert c == self.d_model

        # input projection
        qkv = self.project_qkv(input.view(n * b, self.d_model)).view(n, b * heads, self.d_kqv)

        # split
        q = qkv[:, :, :self.d_k]
        k = qkv[:, :, self.d_k:2 * self.d_k]
        v = qkv[:, :, 2 * self.d_k:]

        # main attention calculation
        # weights: (b*h, n_q, n_k)
        # att_raw: (n_q, b*h, d_v)

        logits = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2))
        weights = nnf.softmax(logits, -1)

        # this contiguous() is not actually necessary if we could set the output strides
        att_raw = torch.bmm(weights, v.transpose(0, 1)).transpose(0, 1).contiguous()

        att_viewed = att_raw.view(n * b, heads * self.d_v)
        att_projected = self.project_out(att_viewed).view(n, b, self.d_model)
        att_result = self.norm_att(input * self.alpha + self.dropout(att_projected))

        ff_inner = self.ff(att_result.view(n * b, self.d_model)).view(n, b, self.d_model)
        ff_result = self.norm_ff(att_result * self.alpha + self.dropout(ff_inner))

        return ff_result, weights

    def forward(self, input):
        result, _ = self.forward_with_weights(input)
        return result


class AttShapes(NamedTuple):
    b: int  # batch size
    n: int  # input sequence length (for k and v)
    m: int  # output sequence length (for q)
    heads: int  # the number of heads
    dk: int  # key size per head (for q and k)
    dv: int  # value size per head


def check_att_shapes(q, k, v, heads: int) -> AttShapes:
    m, b0, dk_total0 = q.shape
    n0, b1, dk_total1 = k.shape
    n1, b2, dv_total = v.shape

    assert b0 == b1 == b2, "Batch size mismatch"
    assert n0 == n1, "Input seq length mismatch"
    assert dk_total0 == dk_total1, "Key size mismatch"

    assert dk_total0 % heads == 0 and dv_total % heads == 0, "Size not divisible by heads"

    return AttShapes(b=b0, n=n0, m=m, heads=heads, dk=dk_total0 // heads, dv=dv_total // heads)
