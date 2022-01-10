from torch import nn


class ResModule(nn.Module):
    def __init__(self, *inner: nn.Module):
        super().__init__()
        self.inner = nn.Sequential(*inner)

    def forward(self, x):
        return self.inner(x) + x
