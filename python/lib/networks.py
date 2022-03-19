from typing import Optional

from torch import nn


class MuZeroNetworks(nn.Module):
    def __init__(
            self,
            state_channels: int, state_quant_bits: Optional[int],
            representation: nn.Module, dynamics: nn.Module, prediction: nn.Module
    ):
        super().__init__()

        self.state_channels = state_channels
        self.state_quant_bits = state_quant_bits

        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
