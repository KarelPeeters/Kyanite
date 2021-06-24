from math import prod

import torch
from matplotlib import pyplot
from torch.optim import AdamW

from models import GoogleModel
from train import TrainSettings, train_model, ValueTarget
from util import load_data, DEVICE, GoogleData


def plot_stuff(plot_data, plot_legend):
    pyplot.plot(plot_data)
    pyplot.legend(plot_legend)
    pyplot.show()


def main():
    all_data = load_data("../data/esat2/all_data.csv")
    train_data = GoogleData.from_generic(all_data.pick_batch(slice(0, 2_000_000))).to(DEVICE)
    test_data = GoogleData.from_generic(all_data.pick_batch(slice(2_000_000, 2_100_000))).to(DEVICE)

    print(f"train size: {len(train_data)}")
    print(f"test size: {len(test_data)}")

    model = GoogleModel(
        channels=32, blocks=8,
        value_channels=1, value_size=16,
        policy_channels=4,
        res=True,
        squeeze_size=None, squeeze_bias=False
    )

    param_count = sum(prod(p.shape) for p in model.parameters())
    print(f"Model has {param_count} parameters, which takes {param_count // 1024 / 1024:.3f} Mb")
    for name, child in model.named_children():
        child_param_count = sum(prod(p.shape) for p in child.parameters())
        print(f"  {name}: {child_param_count / param_count:.2f}")

    model = torch.jit.script(model)
    model.to(DEVICE)

    batch_size = 256
    # cycles_per_epoch = 2

    optimizer = AdamW(model.parameters(), weight_decay=1e-5)
    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=1e-4, max_lr=1e-2,
    #     cycle_momentum=True,
    #     base_momentum=0.8, max_momentum=0.9,
    #     step_size_up=len(train_data) // (batch_size * cycles_per_epoch)
    # )

    settings = TrainSettings(
        output_path="../data/esat/modest",
        train_data=train_data,
        test_data=test_data,
        epochs=5,
        optimizer=optimizer,
        scheduler=None,
        value_target=ValueTarget.FinalValue,
        policy_weight=1.0,
        batch_size=batch_size,
        plot_points=100,
        plot_window_size=5,
    )

    train_model(model, settings)
    # plot_train_data(settings)

    # TODO plot loss(number of times a state appears in the train data)
    #   remove train states from the test set? currently a lot of them are repeating
    #   or maybe this doesn't matter that much? check again how many duplicate states we have


if __name__ == '__main__':
    main()
