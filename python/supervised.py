from math import prod

import torch
from torch.optim import AdamW

from models import MixModel
from train import TrainState, train_model, WdlTarget, TrainSettings
from util import load_data, DEVICE


def print_data_stats(test_data, train_data):
    print(f"train size: {len(train_data)}")
    print(f"test size: {len(test_data)}")

    train_unique_count = len(torch.unique(train_data.input, dim=0))
    test_unique_count = len(torch.unique(test_data.input, dim=0))
    both_unique_count = len(torch.unique(torch.cat([train_data.input, test_data.input], dim=0), dim=0))

    test_unique = test_unique_count / len(test_data)
    test_unique_different = (both_unique_count - train_unique_count) / test_unique_count

    print(f"train unique: {train_unique_count / len(train_data):.3f}")
    print(f"test unique & not in test: {test_unique * test_unique_different:.3f}")


def main():
    # TODO this is completely wrong! we need to keep games separate between test and train sets!
    #   right now we're leaking like crazy!

    # shuffle to avoid biasing test data towards longer games
    # train_data, test_data = load_data("../data/esat3/games.csv", test_fraction=0.02)
    train_data, test_data = load_data("../data/esat3/games.csv", test_fraction=0.02)

    # print_data_stats(test_data, train_data)

    train_data = train_data.to(DEVICE)
    test_data = test_data.to(DEVICE)

    # TODO this is okay for value, but policy is struggling :(
    #   hypothesis: the messages (with relu!) are too small to carry the required information
    #       maybe make messages smaller and instead carry a lot of state within each macro?
    model = MixModel(
        depth=8, message_size=32, wdl_size=32,
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
        epochs=15,
        wdl_target=WdlTarget.Final,
        policy_weight=2.0,
        batch_size=batch_size,
        plot_points=100,
        plot_smooth_points=50,
    )

    state = TrainState(
        settings=settings,
        output_path="../data/esat3/mix_large",
        train_data=train_data,
        test_data=test_data,
        optimizer=optimizer,
        scheduler=None,
    )

    train_model(model, state)
    # plot_train_data(settings)

    # TODO plot loss(number of times a state appears in the train data)
    #   remove train states from the test set? currently a lot of them are repeating
    #   or maybe this doesn't matter that much? check again how many duplicate states we have


if __name__ == '__main__':
    main()
