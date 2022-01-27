import glob
import os
import re
from typing import Optional

import torch
from torch.optim import SGD

from lib.data.buffer import FileListSampler
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.model.post_act import PostActNetwork, PostActScalarHead, PostActAttentionPolicyHead
from lib.plotter import LogPlotter, run_with_plotter
from lib.schedule import FixedSchedule, WarmupSchedule
from lib.supervised import supervised_loop
from lib.train import TrainSettings, ScalarTarget
from lib.util import DEVICE, print_param_count


def find_last_finished_batch(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None

    last_finished = -1
    for file in os.listdir(path):
        m = re.match(r"network_(\d+).onnx", file)
        if m:
            last_finished = max(last_finished, int(m.group(1)))

    return last_finished if last_finished >= 0 else None


def main(plotter: LogPlotter):
    output_folder = "../../data/supervised/moves_left/added"

    train_pattern = "../../data/loop/chess/simple_unbalanced/selfplay/*.json"
    test_pattern = "../../data/loop/chess/simple_unbalanced/selfplay/*.json"
    limit_file_count = 100

    game = Game.find("chess")
    os.makedirs(output_folder, exist_ok=True)
    allow_resume = False

    batch_size = 128

    test_steps = 16
    save_steps = 128

    settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=1.0,
        policy_weight=1.0,
        moves_left_delta=20,
        moves_left_weight=0.0001,
        clip_norm=20.0,
        value_target=ScalarTarget.Final,
        train_in_eval_mode=False,
    )

    def initial_network():
        old_network = torch.jit.load(
            "C:/Documents/Programming/STTT/AlphaZero/data/loop/chess/simple_unbalanced/training/gen_462/network.pt")
        channels = 32
        new_network = PostActNetwork(
            game, 8, channels,
            PostActScalarHead(game, channels, 4, 32),
            PostActAttentionPolicyHead(game, channels, channels),
        )

        old_params = list(old_network.named_parameters())
        new_params = list(new_network.named_parameters())
        assert len(old_params) == len(new_params)

        for (new_name, new_param), (old_name, old_param) in zip(new_params, old_params):
            old_name = old_name.replace("value_head", "scalar_head")
            assert new_name == old_name, f"Name mismatch: {new_name} vs {old_name}"

            if new_name == old_name:
                if new_param.shape == old_param.shape:
                    new_param.data.copy_(old_param)
                else:
                    print(f"Skipping shape mismatch {new_name}: new {new_param.shape} old {old_param.shape}")

        return new_network

    train_files = sorted((DataFile.open(game, p) for p in glob.glob(train_pattern)), key=lambda f: f.info.timestamp)
    test_files = sorted((DataFile.open(game, p) for p in glob.glob(test_pattern)), key=lambda f: f.info.timestamp)

    if limit_file_count is not None:
        train_files = train_files[-min(limit_file_count, len(train_files)):]
        test_files = test_files[-min(limit_file_count, len(train_files)):]

    train_sampler = FileListSampler(game, train_files, batch_size)
    test_sampler = FileListSampler(game, test_files, batch_size)

    print(f"Train file count: {len(train_files)}")
    print(f"Train file game count: {sum(f.info.game_count for f in train_files)}")
    print(f"Train position count: {len(train_sampler)}")

    print(f"Test file count: {len(test_files)}")
    print(f"Train file game count: {sum(f.info.game_count for f in test_files)}")
    print(f"Test position count: {len(test_sampler)}")

    last_bi = find_last_finished_batch(output_folder)

    if last_bi is None:
        logger = Logger()
        start_bi = 0
        network = initial_network()
    else:
        assert allow_resume, f"Not allowed to resume, but found existing batch {last_bi}"

        logger = Logger.load(os.path.join(output_folder, "log.npz"))
        start_bi = last_bi + 1
        network = torch.jit.load(os.path.join(output_folder, f"network_{last_bi}.pb"))

    network.to(DEVICE)
    print_param_count(network)

    optimizer = SGD(network.parameters(), weight_decay=1e-5, lr=0.0, momentum=0.9)
    schedule = WarmupSchedule(100, FixedSchedule([0.02, 0.01, 0.001], [900, 2_000]))

    plotter.set_title(f"supervised {output_folder}")
    plotter.set_can_pause(True)
    plotter.update(logger)

    supervised_loop(
        settings, schedule, optimizer,
        start_bi, output_folder, logger, plotter,
        network, train_sampler, test_sampler,
        test_steps, save_steps,
    )

    # currently, these never trigger (since the loop never stops), but that may change in the future
    train_sampler.close()
    test_sampler.close()


if __name__ == '__main__':
    run_with_plotter(main)
