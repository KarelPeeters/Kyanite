import os
from math import prod

import torch
from matplotlib import pyplot
from torch.optim import AdamW

from core import train_model
from models import GoogleModel
from util import load_data, DEVICE, GoogleData


def plot_stuff(plot_data, plot_legend):
    pyplot.plot(plot_data)
    pyplot.legend(plot_legend)
    pyplot.show()


def main():
    train_data = GoogleData.from_generic(load_data("../data/esat/train_data.csv")).to(DEVICE)
    test_data = GoogleData.from_generic(load_data("../data/esat/test_data.csv")).to(DEVICE)

    print(f"Train size: {len(train_data)}, test size: {len(test_data)}")

    OUTPUT_FOLDER = "../data/esat/deeper"
    EPOCHS = 5

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    model = GoogleModel(channels=32, blocks=8, value_channels=8, value_size=32, policy_channels=8, res=True)
    model.to(DEVICE)

    param_count = sum(prod(p.shape) for p in model.parameters())
    print(f"Model has {param_count} parameters, which takes {param_count // 1024 / 1024:.3f} Mb")

    model = torch.jit.script(model)
    model.save(f"{OUTPUT_FOLDER}/{0}_epochs.pt")

    optimizer = AdamW(model.parameters(), weight_decay=1e-5)

    for e in range(EPOCHS):
        _, _ = train_model(
            model=model, optimizer=optimizer,
            policy_weight=1,
            train_data=train_data, test_data=test_data,
            epochs=1, train_batch_size=256,
            eval_batch_size=128,
        )

        model.save(f"{OUTPUT_FOLDER}/{e + 1}_epochs.pt")


if __name__ == '__main__':
    main()
