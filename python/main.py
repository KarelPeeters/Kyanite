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
            nn.Linear(2 * 81 + 2 * 9, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, data: Data):
        x = torch.cat([
            data.x_tiles.flatten(start_dim=1),
            data.x_macros.flatten(start_dim=1)
        ], dim=1)

        return self.seq(x)


def main():
    train_data = load_data("../data/train_data.csv", "../data/train_data.pt")
    test_data = load_data("../data/test_data.csv", "../data/test_data.pt")

    print(f"Train size: {len(train_data)}, test size: {len(test_data)}")

    trivial_accuracy = train_data.y_win.float().mean(dim=0).max()
    print(f"Trivial accuracy: {trivial_accuracy}")

    model = Model()
    model.to(DEVICE)

    loss_func = nn.MSELoss()
    optimizer = Adam(model.parameters())

    plot_data, plot_legend = train_model(
        model=model, optimizer=optimizer, loss_func=loss_func,
        train_data=train_data, test_data=test_data,
        epochs=30,
    )

    pyplot.plot(plot_data)
    pyplot.axhline(y=trivial_accuracy, linestyle="--")
    pyplot.legend(plot_legend + ["trivial_acc"])
    pyplot.show()


if __name__ == '__main__':
    main()
