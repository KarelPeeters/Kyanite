from typing import Optional

from torch import nn


class MuZeroNetworks(nn.Module):
    def __init__(
            self,
            state_channels: int, state_quant_bits: Optional[int], state_channels_saved: int,
            representation: nn.Module, dynamics: nn.Module, prediction: nn.Module
    ):
        assert state_channels <= state_channels
        super().__init__()

        self.state_channels = state_channels
        self.state_channels_saved = state_channels_saved

        self.state_quant_bits = state_quant_bits

        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
