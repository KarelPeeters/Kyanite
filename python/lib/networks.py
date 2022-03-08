from torch import nn


class MuZeroNetworks(nn.Module):
    def __init__(self, state_channels: int, representation: nn.Module, dynamics: nn.Module, prediction: nn.Module):
        super().__init__()

        self.state_channels = state_channels

        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
