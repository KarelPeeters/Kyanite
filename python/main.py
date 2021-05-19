from math import prod

import torch
from matplotlib import pyplot
from torch.optim import Adam

from core import train_model
from models import GoogleModel
from util import load_data, DEVICE


def plot_stuff(plot_data, plot_legend):
    pyplot.plot(plot_data)
    pyplot.legend(plot_legend)
    pyplot.show()


def main():
    train_data = load_data("../data/esat/train_data.csv")
    test_data = load_data("../data/test_data.csv")

    print(f"Train size: {len(train_data)}, test size: {len(test_data)}")

    trivial_win_acc = train_data.y_win.float().mean(dim=0).max()
    trivial_move_acc = train_data.y_move_prob.mean(dim=0).max()

    print(f"Trivial win_acc: {trivial_win_acc}")
    print(f"Trivial move_acc: {trivial_move_acc}")

    model = GoogleModel(channels=64, block_count=3, value_size=64)
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
