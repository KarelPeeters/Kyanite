import itertools
import os
from typing import Callable

import torch.jit
from torch import nn
from torch.optim import AdamW, Optimizer

from loop import LoopSettings
from models import TowerModel, ResBlock
from plot_loop import plot_loops
from train import train_model, TrainState, TrainSettings, WdlTarget
from util import DEVICE, print_param_count


def retrain(
        model: nn.Module,
        prev_path: str,
        prev_offset: int,
        new_path: str,
        settings: LoopSettings,
        recreate_optimizer: bool,
        optimizer: Callable[[nn.Module, float], Optimizer],
):
    train_path = os.path.join(new_path, "training")
    os.makedirs(train_path, exist_ok=False)

    buffer = settings.new_buffer()
    curr_optimizer = optimizer(model, settings.train_weight_decay)

    for gi in itertools.count():
        games_path = os.path.join(prev_path, "selfplay", f"games_{gi + prev_offset}.bin")
        print(f"Trying to load {games_path}")
        if not os.path.exists(games_path):
            break
        buffer.push_load_path(games_path)

        if recreate_optimizer:
            curr_optimizer = optimizer(model, settings.train_weight_decay)

        state = TrainState(
            settings.train_settings,
            os.path.join(train_path, f"gen_{gi}"),
            buffer.train_data,
            buffer.test_data,
            curr_optimizer,
            None
        )
        train_model(model, state)

        if gi != 0 and gi % 10 == 0:
            plot_loops([prev_path, new_path], average=True, offsets=[prev_offset, 0])


def main():
    prev_path = "../data/ataxx/test_loop/"
    new_path = "../data/derp/retrain_other/"

    depth = 8
    channels = 32
    inner_channels = 32

    def res_block():
        return ResBlock(channels, inner_channels, True, False, None)

    model = TowerModel(channels, depth, 16, True, True, True, res_block)

    model = torch.jit.script(model)
    model.to(DEVICE)

    print_param_count(model)

    train_settings = TrainSettings(
        epochs=1,
        wdl_target=WdlTarget.Final,
        policy_weight=2.0,
        batch_size=128,
        plot=False,
        plot_points=100,
        plot_smooth_points=100,
    )

    settings = LoopSettings(
        root_path="",
        initial_network=None,
        buffer_gen_count=1,
        test_fraction=0.05,
        fixed_settings=None,
        selfplay_settings=None,
        train_settings=train_settings,
        train_weight_decay=0.0,
    )

    retrain(
        model,
        prev_path,
        100,
        new_path,
        settings,
        False,
        lambda net, decay: AdamW(net.parameters(), weight_decay=decay)
    )


if __name__ == '__main__':
    main()
