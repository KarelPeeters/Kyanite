import torch
from matplotlib import pyplot
from torch import nn
from torch.optim import Adam

from core import train_model
from util import load_data, DEVICE, Data


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(81 + 2 * 81 + 2 * 9, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 1 + 81),
        )

    def forward(self, data: Data):
        input = torch.cat([
            data.mask.flatten(start_dim=1),
            data.x_tiles.flatten(start_dim=1),
            data.x_macros.flatten(start_dim=1),
        ], dim=1)

        output = self.seq(input)

        value = nn.functional.tanh(output[:, 0])
        move_prob = nn.functional.softmax(output[:, 1:], -1)

        return value, move_prob


def plot_stuff(plot_data, plot_legend, trivial_win_acc, trivial_move_acc, plot_loss: bool):
    plot_mask = torch.tensor(["loss" not in name for name in plot_legend])
    plot_mask[:] |= plot_loss
    plot_legend = [name for name in plot_legend if plot_loss or "loss" not in name]

    pyplot.axhline(y=trivial_win_acc, linestyle="--")
    pyplot.axhline(y=trivial_move_acc, linestyle="--")

    pyplot.plot(plot_data[:, plot_mask])
    pyplot.legend(["trivial_acc"] + plot_legend)
    pyplot.show()


def main():
    train_data = load_data("../data/train_data.csv", "../data/train_data.pt")
    test_data = load_data("../data/test_data.csv", "../data/test_data.pt")

    print(f"Train size: {len(train_data)}, test size: {len(test_data)}")

    trivial_win_acc = train_data.y_win.float().mean(dim=0).max()
    trivial_move_acc = train_data.y_move_prob.mean()

    print(f"Trivial win_acc: {trivial_win_acc}")
    print(f"Trivial move_acc: {trivial_move_acc}")

    model = Model()
    model.to(DEVICE)

    optimizer = Adam(model.parameters())

    plot_data, plot_legend = train_model(
        model=model, optimizer=optimizer,
        train_data=train_data, test_data=test_data,
        epochs=50,
    )

    plot_stuff(
        plot_data, plot_legend,
        trivial_win_acc, trivial_move_acc,
        plot_loss=False
    )


if __name__ == '__main__':
    main()
