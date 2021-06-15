import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot
from torch import nn
from torch.optim import Optimizer

from util import DEVICE, GoogleData, linspace_int, uniform_window_filter


def cross_entropy_masked(logits, target, mask):
    assert len(logits.shape) == 2
    assert logits.shape == target.shape

    log = torch.log_softmax(logits + mask.log(), dim=1)
    loss = -(target * log).nansum(dim=1)

    assert not loss.isinf().any(), \
        "inf values in policy loss, maybe the mask and target contain impossible combinations?"

    # average over batch dimension
    return loss.mean(dim=0)


def evaluate_model(model, data: GoogleData):
    # TODO try predicting win/loss/draw again
    #   maybe also use those to predict a value and add that as an 3rd term to the loss
    value_pred, policy_logit = model(data.input)

    value_loss = nn.functional.mse_loss(value_pred, data.value)
    move_loss = cross_entropy_masked(policy_logit, data.policy, data.mask.view(-1, 81))

    return value_loss, move_loss


@dataclass
class TrainSettings:
    output_path: str

    train_data: GoogleData
    test_data: GoogleData

    epochs: int
    optimizer: Optimizer
    scheduler: Any
    policy_weight: float
    batch_size: int

    # the number of plot points per epoch, for both test and train data
    plot_points: int
    plot_window_size: int


def train_model_epoch(model: nn.Module, s: TrainSettings) -> (np.array, np.array):
    batch_size = s.batch_size
    batch_count = len(s.train_data) // batch_size

    plot_batches = linspace_int(batch_count, s.plot_points)
    plot_data = torch.full((len(plot_batches), 7), np.nan, device=DEVICE)
    next_plot_i = 0

    train_shuffle = torch.randperm(len(s.train_data), device=DEVICE)

    for bi in range(batch_count):
        is_plot_batch = bi in plot_batches

        train_batch_i = train_shuffle[bi * batch_size:(bi + 1) * batch_size]
        train_data_batch = s.train_data.pick_batch(train_batch_i)

        if is_plot_batch:
            model.eval()
            test_batch_i = torch.randint(len(s.test_data), (batch_size,), device=DEVICE)
            test_data_batch = s.test_data.pick_batch(test_batch_i)

            test_value_loss, test_policy_loss = evaluate_model(model, test_data_batch)
            test_loss = test_value_loss + s.policy_weight * test_policy_loss
            plot_data[next_plot_i, 3:6] = torch.tensor([test_loss, test_value_loss, test_policy_loss], device=DEVICE)

            print(f"Test batch: {test_loss:.2f}, {test_value_loss:.2f}, {test_policy_loss:.2f}")

        model.train()
        train_value_loss, train_policy_loss = evaluate_model(model, train_data_batch)
        train_loss = train_value_loss + s.policy_weight * train_policy_loss

        if is_plot_batch:
            plot_data[next_plot_i, 0:3] = torch.tensor([train_loss, train_value_loss, train_policy_loss], device=DEVICE)

            if s.scheduler is not None:
                plot_data[next_plot_i, 6] = s.scheduler.get_last_lr()[0]

            next_plot_i += 1

        print(f"Train batch {bi}/{batch_count}: {train_loss:.2f}, {train_value_loss:.2f}, {train_policy_loss:.2f}")

        s.optimizer.zero_grad()
        train_loss.backward()
        s.optimizer.step()

        if s.scheduler is not None:
            s.scheduler.step()

    return plot_data.cpu().numpy()


TRAIN_PLOT_TITLES = ["total", "value", "policy"]
TRAIN_PLOT_LEGEND = ["train", "test"]


def plot_train_data(s: TrainSettings):
    output_path = s.output_path

    all_plot_data = np.load(f"{output_path}/plot_data.npy")
    all_plot_axis = np.load(f"{output_path}/plot_axis.npy")

    fig, axes = pyplot.subplots(4)
    for i in range(3):
        ax = axes[i]

        train_smooth_values = uniform_window_filter(all_plot_data[:, i], s.plot_window_size)
        ax.plot(all_plot_axis, train_smooth_values)

        test_smooth_values = uniform_window_filter(all_plot_data[:, 3 + i], s.plot_window_size)
        ax.plot(all_plot_axis, test_smooth_values)

        ax.set_title(TRAIN_PLOT_TITLES[i])

    ax = axes[3]
    ax.plot(all_plot_axis, all_plot_data[:, 6])
    ax.set_title("learning_rate")

    fig.legend(TRAIN_PLOT_LEGEND)
    fig.suptitle(f"Training progress")
    fig.tight_layout()

    fig.savefig(f"{output_path}/plot.png")
    fig.show()


def train_model(model: nn.Module, s: TrainSettings):
    epochs = s.epochs
    output_path = s.output_path

    all_plot_data = None

    os.makedirs(s.output_path, exist_ok=True)
    model.save(f"{s.output_path}/{0}_epochs.pt")

    for ei in range(epochs):
        print(f"Starting epoch {ei + 1}/{epochs}")

        plot_data = train_model_epoch(model, s)
        model.save(f"{output_path}/model_{ei + 1}_epochs.pt")

        if all_plot_data is None:
            all_plot_data = plot_data
        else:
            all_plot_data = np.concatenate((all_plot_data, plot_data), axis=0)

        all_plot_axis = np.linspace(0, ei + 1, endpoint=False, num=len(all_plot_data))

        np.save(f"{output_path}/plot_data.npy", all_plot_data)
        np.save(f"{output_path}/plot_axis.npy", all_plot_axis)

        plot_train_data(s)
