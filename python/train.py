import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot
from torch import nn
from torch.optim import Optimizer

from util import DEVICE, linspace_int, uniform_window_filter, GameData


def cross_entropy_masked(logits, target, mask):
    assert len(logits.shape) == 2, logits.shape
    assert logits.shape == target.shape, (logits.shape, target.shape)
    assert mask is None or logits.shape == mask.shape, (logits.shape, mask.shape)

    # ignore empty masks
    if mask is not None:
        mask_empty = mask.sum(dim=1) == 0
        logits = logits[~mask_empty, :]
        target = target[~mask_empty, :]
        mask = mask[~mask_empty, :]

    # the mechanism for ignoring masked values:
    #   log converts 0->-inf, 1->0
    #   log_softmax converts -inf->nan
    #   nansum then ignores these propagated nan values
    mask_log = mask.log() if mask is not None else 0
    log = torch.log_softmax(logits + mask_log, dim=1)
    loss = -(target * log).nansum(dim=1)

    assert loss.isfinite().all(), \
        "inf/nan values in loss, maybe the mask and target contain impossible combinations?"

    # average over batch dimension
    return loss.mean(dim=0)


def evaluate_model(model, data: GameData, target: 'WdlTarget'):
    wdl_logit, policy_logit = model(data.board)

    wdl_target = target.get_target(data.wdl_final, data.wdl_est)
    wdl_loss_ce = cross_entropy_masked(wdl_logit, wdl_target, None)
    wdl_loss_mse = nn.functional.mse_loss(nn.functional.softmax(wdl_logit, -1), wdl_target)

    policy_loss = cross_entropy_masked(
        policy_logit.flatten(start_dim=1),
        data.policy.flatten(start_dim=1),
        data.policy_mask.flatten(start_dim=1)
    )

    return wdl_loss_ce, wdl_loss_mse, policy_loss


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


class WdlLoss(Enum):
    CrossEntropy = auto()
    MSE = auto()

    def select(self, ce, mse):
        if self == WdlLoss.CrossEntropy:
            return ce
        if self == WdlLoss.MSE:
            return mse
        assert False, self


@dataclass
class TrainSettings:
    epochs: int
    wdl_target: WdlTarget
    wdl_loss: WdlLoss
    policy_weight: float
    batch_size: int

    plot: bool
    # the number of plot points per epoch, for both test and train data
    plot_points: int
    plot_smooth_points: int


@dataclass
class TrainState:
    settings: TrainSettings

    output_path: str

    train_data: GameData
    test_data: GameData

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

        train_batch_i = train_shuffle[bi * batch_size:(bi + 1) * batch_size]
        train_data_batch = s.train_data[train_batch_i].random_symmetry()

        if is_plot_batch:
            model.eval()
            test_batch_i = torch.randint(len(s.test_data), (batch_size,), device=DEVICE)
            test_data_batch = s.test_data[test_batch_i].random_symmetry()

            test_value_loss_ce, test_value_loss_mse, test_policy_loss = \
                evaluate_model(model, test_data_batch, s.settings.wdl_target)
            test_value_loss = s.settings.wdl_loss.select(test_value_loss_ce, test_value_loss_mse)

            test_loss = test_value_loss + s.settings.policy_weight * test_policy_loss
            plot_data[next_plot_i, 3:6] = torch.tensor([test_loss, test_value_loss, test_policy_loss], device=DEVICE)

            print(
                f"Test batch: {test_loss:.2f} = {test_value_loss:.2f} + c * {test_policy_loss:.2f},"
                f" note: ce={test_value_loss_ce}, mse={test_value_loss_mse}"
            )

        model.train()
        train_value_loss_ce, train_value_loss_mse, train_policy_loss = \
            evaluate_model(model, train_data_batch, s.settings.wdl_target)
        train_value_loss = s.settings.wdl_loss.select(train_value_loss_ce, train_value_loss_mse)

        train_loss = train_value_loss + s.settings.policy_weight * train_policy_loss

        if is_plot_batch:
            plot_data[next_plot_i, 0:3] = torch.tensor([train_loss, train_value_loss, train_policy_loss], device=DEVICE)

            if s.scheduler is not None:
                plot_data[next_plot_i, 6] = s.scheduler.get_last_lr()[0]

            next_plot_i += 1

        print(
            f"Epoch {ei + 1}, train batch {bi}/{batch_count}: {train_loss:.2f} ="
            f" {train_value_loss:.2f} + c * {train_policy_loss:.2f},"
            f" note: ce={train_value_loss_ce}, mse={train_value_loss_mse}"
        )

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

        train_smooth_values = uniform_window_filter(all_plot_data[:, i], smooth_window_size, 0)
        pyplot.plot(all_plot_axis, train_smooth_values, label="train")

        test_smooth_values = uniform_window_filter(all_plot_data[:, 3 + i], smooth_window_size, 0)
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


def save_onnx(network, onnx_path: str):
    print(f"Saving model to {onnx_path}")
    network.eval()
    example_input = torch.zeros(1, 3, 7, 7, device=DEVICE)
    example_outputs = network(example_input)
    torch.onnx.export(
        model=network,
        args=example_input,
        f=onnx_path,
        example_outputs=example_outputs,
        input_names=["input"],
        output_names=["wdl", "policy"],
        dynamic_axes={"input": {0: "batch_size"}, "wdl": {0: "batch_size"}, "policy": {0: "batch_size"}},
    )


def train_model(model: nn.Module, s: TrainState):
    epochs = s.settings.epochs
    output_path = s.output_path

    all_plot_data = None

    os.makedirs(s.output_path, exist_ok=True)
    torch.jit.save(model, f"{output_path}/model_{0}_epochs.pt")
    save_onnx(model, f"{output_path}/model_{0}_epochs.onnx")

    for ei in range(epochs):
        print(f"Starting epoch {ei + 1}/{epochs}")

        plot_data = train_model_epoch(ei, model, s)
        torch.jit.save(model, f"{output_path}/model_{ei + 1}_epochs.pt")
        save_onnx(model, f"{output_path}/model_{ei + 1}_epochs.onnx")

        if all_plot_data is None:
            all_plot_data = plot_data
        else:
            all_plot_data = np.concatenate((all_plot_data, plot_data), axis=0)

        all_plot_axis = np.linspace(0, ei + 1, endpoint=False, num=len(all_plot_data))

        np.save(f"{output_path}/plot_data.npy", all_plot_data)
        np.save(f"{output_path}/plot_axis.npy", all_plot_axis)

        if s.settings.plot:
            plot_train_data(s)
