import itertools
import os
from typing import Callable

import torch.jit
from torch import nn
from torch.optim import AdamW, Optimizer

from loop import LoopSettings
from models import TowerModel, ResBlock
from train import train_model, TrainState, TrainSettings, WdlTarget
from util import DEVICE


def retrain(
        model: nn.Module,
        selfplay_path: str,
        temp_path: str,
        settings: LoopSettings,
        recreate_optimizer: bool,
        optimizer: Callable[[nn.Module, float], Optimizer],
):
    train_path = os.path.join(temp_path, "training")
    os.makedirs(train_path, exist_ok=True)

    buffer = settings.new_buffer()
    curr_optimizer = optimizer(model, settings.train_weight_decay)

    for gi in itertools.count():
        games_path = os.path.join(selfplay_path, f"games_{gi}.bin")
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


def main():
    selfplay_path = "../data/ataxx/test_loop/selfplay"
    temp_path = "../data/derp/retrain_other/"

    model = TowerModel(32, 8, 16, True, True, True, lambda: ResBlock(32, 32, True, True, None))
    model = torch.jit.script(model)
    model.to(DEVICE)

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
        train_weight_decay=1e-6,
    )

    retrain(
        model,
        selfplay_path,
        temp_path,
        settings,
        False,
        lambda net, decay: AdamW(net.parameters(), weight_decay=decay)
    )


if __name__ == '__main__':
    main()
