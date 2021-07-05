import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot
from torch import nn
from torch.optim import Optimizer

from util import DEVICE, GoogleData, linspace_int, uniform_window_filter, GenericData


def cross_entropy_masked(logits, target, mask):
    assert len(logits.shape) == 2
    assert logits.shape == target.shape

    mask_log = mask.log() if mask is not None else 0
    log = torch.log_softmax(logits + mask_log, dim=1)
    loss = -(target * log).nansum(dim=1)

    assert not loss.isinf().any(), \
        "inf values in loss, maybe the mask and target contain impossible combinations?"

    # average over batch dimension
    return loss.mean(dim=0)


def evaluate_model(model, data: GenericData, target: 'WdlTarget'):
    input = torch.cat([
        data.tiles_o.view(-1, 2, 9, 9),
        data.macros.view(-1, 2, 1, 9),
    ], dim=2)
    wdl_logit, policy_logit = model(data.mask_o.view(-1, 9, 9), input)

    wdl_target = target.get_target(data.wdl_final, data.wdl_est)
    wdl_loss = cross_entropy_masked(wdl_logit, wdl_target, None)
    move_loss = cross_entropy_masked(policy_logit, data.policy_o, data.mask_o)

    return wdl_loss, move_loss


class WdlTarget(Enum):
    Final = auto()
    Estimate = auto()
    Mean = auto()

    def get_target(self, final, est):
        if self == WdlTarget.Final:
            return final
        if self == WdlTarget.Estimate:
            return est
        if self == WdlTarget.Mean:
            return (final + est) / 2
        assert False, self


@dataclass
class TrainSettings:
    epochs: int
    wdl_target: WdlTarget
    policy_weight: float
    batch_size: int

    # the number of plot points per epoch, for both test and train data
    plot_points: int
    plot_smooth_points: int


@dataclass
class TrainState:
    settings: TrainSettings

    output_path: str

    train_data: GoogleData
    test_data: GoogleData

    optimizer: Optimizer
    scheduler: Any


def train_model_epoch(ei: int, model: nn.Module, s: TrainState) -> (np.array, np.array):
    batch_size = s.settings.batch_size
    batch_count = len(s.train_data) // batch_size

    plot_batches = linspace_int(batch_count, s.settings.plot_points)
    plot_data = torch.full((len(plot_batches), 7), np.nan, device=DEVICE)
    next_plot_i = 0

    train_shuffle = torch.randperm(len(s.train_data), device=DEVICE)

    for bi in range(batch_count):
        is_plot_batch = bi in plot_batches

        # todo bring random symmetry back

        train_batch_i = train_shuffle[bi * batch_size:(bi + 1) * batch_size]
        train_data_batch = s.train_data.pick_batch(train_batch_i)  # .random_symmetry()

        if is_plot_batch:
            model.eval()
            test_batch_i = torch.randint(len(s.test_data), (batch_size,), device=DEVICE)
            test_data_batch = s.test_data.pick_batch(test_batch_i)  # .random_symmetry()

            test_value_loss, test_policy_loss = evaluate_model(model, test_data_batch, s.settings.wdl_target)
            test_loss = test_value_loss + s.settings.policy_weight * test_policy_loss
            plot_data[next_plot_i, 3:6] = torch.tensor([test_loss, test_value_loss, test_policy_loss], device=DEVICE)

            print(f"Test batch: {test_loss:.2f}, {test_value_loss:.2f}, {test_policy_loss:.2f}")

        model.train()
        train_value_loss, train_policy_loss = evaluate_model(model, train_data_batch, s.settings.wdl_target)
        train_loss = train_value_loss + s.settings.policy_weight * train_policy_loss

        if is_plot_batch:
            plot_data[next_plot_i, 0:3] = torch.tensor([train_loss, train_value_loss, train_policy_loss], device=DEVICE)

            if s.scheduler is not None:
                plot_data[next_plot_i, 6] = s.scheduler.get_last_lr()[0]

            next_plot_i += 1

        print(
            f"Epoch {ei + 1}, train batch {bi}/{batch_count}: {train_loss:.2f},"
            f" {train_value_loss:.2f}, {train_policy_loss:.2f}")

        s.optimizer.zero_grad()
        train_loss.backward()
        s.optimizer.step()

        if s.scheduler is not None:
            s.scheduler.step()

    return plot_data.cpu().numpy()


TRAIN_PLOT_TITLES = ["total", "value", "policy"]
TRAIN_PLOT_LEGEND = ["train", "test"]


def plot_train_data(s: TrainState):
    output_path = s.output_path

    all_plot_data = np.load(f"{output_path}/plot_data.npy")
    all_plot_axis = np.load(f"{output_path}/plot_axis.npy")

    has_schedule = s.scheduler is not None

    for i in range(3):
        pyplot.figure()

        smooth_window_size = int(len(all_plot_data) / s.settings.plot_smooth_points) + 1

        train_smooth_values = uniform_window_filter(all_plot_data[:, i], smooth_window_size)
        pyplot.plot(all_plot_axis, train_smooth_values, label="train")

        test_smooth_values = uniform_window_filter(all_plot_data[:, 3 + i], smooth_window_size)
        pyplot.plot(all_plot_axis, test_smooth_values, label="test")

        pyplot.title(TRAIN_PLOT_TITLES[i])
        pyplot.legend()

        pyplot.savefig(f"{output_path}/plot_{TRAIN_PLOT_TITLES[i]}.png")
        pyplot.show()

    if has_schedule:
        pyplot.figure()
        pyplot.plot(all_plot_axis, all_plot_data[:, 6])
        pyplot.title("Learning rate schedule")

        pyplot.savefig(f"{output_path}/plot_lr_schedule.png")
        pyplot.show()


def train_model(model: nn.Module, s: TrainState):
    epochs = s.settings.epochs
    output_path = s.output_path

    all_plot_data = None

    os.makedirs(s.output_path, exist_ok=True)
    model.save(f"{s.output_path}/model_0_epochs.pt")

    for ei in range(epochs):
        print(f"Starting epoch {ei + 1}/{epochs}")

        plot_data = train_model_epoch(ei, model, s)
        model.save(f"{output_path}/model_{ei + 1}_epochs.pt")

        if all_plot_data is None:
            all_plot_data = plot_data
        else:
            all_plot_data = np.concatenate((all_plot_data, plot_data), axis=0)

        all_plot_axis = np.linspace(0, ei + 1, endpoint=False, num=len(all_plot_data))

        np.save(f"{output_path}/plot_data.npy", all_plot_data)
        np.save(f"{output_path}/plot_axis.npy", all_plot_axis)

        plot_train_data(s)
