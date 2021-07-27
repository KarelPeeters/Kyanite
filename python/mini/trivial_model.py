import torch
from torch import nn

from util import save_fused_params


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, input):
        return (self.conv(input) + input)#.relu()


def main():
    torch.random.manual_seed(99)

    model = MyModel()
    save_fused_params(model, "../../data/derp/params.npz")

    input = torch.randn(1, 1, 1, 1)

    output = model(input)

    input_flat = list(input.numpy().flatten())
    output_flat = list(output.detach().numpy().flatten())

    print(input.shape)
    print(input_flat)
    print(output.shape)
    print(output_flat)


if __name__ == '__main__':
    main()
