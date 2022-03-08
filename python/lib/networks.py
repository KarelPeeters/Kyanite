from torch import nn


class MuZeroNetworks(nn.Module):
    def __init__(self, representation: nn.Module, dynamics: nn.Module, prediction: nn.Module):
        super().__init__()
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction


if __name__ == '__main__':
    main()
