from math import prod

import torch
from matplotlib import pyplot
from torch import nn
from torch.optim import Adam

from core import train_model
from util import load_data, DEVICE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        y = self.seq(x)
        y = torch.relu(y + x)
        return y


class Model(nn.Module):
    def __init__(self, channels: int, block_count: int, value_size: int):
        super().__init__()

        self.common = nn.Sequential(
            nn.Conv2d(5, channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *(ResBlock(channels) for _ in range(block_count))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, (1, 1)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 9 * 9),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, (1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * 9, value_size),
            nn.ReLU(),
            nn.Linear(value_size, 1),
            nn.Tanh(),
        )

    def forward(self, mask, x_tiles, x_macros):
        x_macros_expanded = torch.kron(x_macros, torch.ones(3, 3, device=x_macros.device))

        input = torch.cat([
            mask.view(-1, 1, 9, 9),
            x_tiles.view(-1, 2, 9, 9),
            x_macros_expanded.view(-1, 2, 9, 9),
        ], dim=1)

        common = self.common(input)
        value = self.value_head(common).squeeze(dim=1)
        policy = self.policy_head(common)

        # value is in range -1..1, policy are the logits
        return value, policy


def plot_stuff(plot_data, plot_legend):
    pyplot.plot(plot_data)
    pyplot.legend(plot_legend)
    pyplot.show()


def main():
    train_data = load_data("../data/train_data.csv", "../data/train_data.pt")
    test_data = load_data("../data/test_data.csv", "../data/test_data.pt")

    print(f"Train size: {len(train_data)}, test size: {len(test_data)}")

    trivial_win_acc = train_data.y_win.float().mean(dim=0).max()
    trivial_move_acc = train_data.y_move_prob.mean(dim=0).max()

    print(f"Trivial win_acc: {trivial_win_acc}")
    print(f"Trivial move_acc: {trivial_move_acc}")

    model = Model(channels=64, block_count=3, value_size=64)
    model.to(DEVICE)

    model = torch.jit.script(model)
    model.save("../data/untrained_model.pt")

    param_count = sum(prod(p.shape) for p in model.parameters())
    print(f"Model has {param_count} parameters, which takes {param_count // 1024 / 1024:.3f} Mb")

    optimizer = Adam(model.parameters(), weight_decay=1e-4)

    plot_data, plot_legend = train_model(
        model=model, optimizer=optimizer,
        train_data=train_data, test_data=test_data,
        epochs=5, train_batch_size=300,
        eval_batch_size=300,
    )

    model.save("../data/trained_model.pt")

    plot_stuff(plot_data, plot_legend)


if __name__ == '__main__':
    main()
