import os
import re
from typing import Optional

import torch

from lib.data.file import DataFile
from lib.data.file_list import FileListSampler, FileList
from lib.games import Game
from lib.logger import Logger
from lib.model.attention import AttentionTower
from lib.model.post_act import ScalarHead, AttentionPolicyHead, PredictionHeads
from lib.plotter import LogPlotter, run_with_plotter
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
    output_folder = "../../data/supervised/att-again-deeper"

    paths = [
        fr"C:\Documents\Programming\STTT\AlphaZero\data\loop\chess\16x128\selfplay\games_{i}.bin"
        for i in range(2600, 3600)
    ]

    train_paths = paths
    test_paths = paths
    limit_file_count: Optional[int] = None

    game = Game.find("chess")
    os.makedirs(output_folder, exist_ok=True)
    allow_resume = True

    batch_size = 256

    test_steps = 16
    save_steps = 128

    settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=1.0,
        policy_weight=1.0,
        moves_left_delta=20,
        moves_left_weight=0.0001,
        clip_norm=5.0,
        scalar_target=ScalarTarget.Final,
        train_in_eval_mode=False,
        mask_policy=True,
    )
    include_final: bool = False

    def initial_network():
        channels = 256
        return PredictionHeads(
            common=AttentionTower(game.board_size, game.full_input_channels, 16, channels, 8, 16, 16, 256, 0.1),
            # common=ResTower(8, game.full_input_channels, channels),

            scalar_head=ScalarHead(game.board_size, channels, 4, 32),
            policy_head=AttentionPolicyHead(game, channels, channels),
        )

    train_files = sorted((DataFile.open(game, p) for p in train_paths), key=lambda f: f.info.timestamp)
    test_files = sorted((DataFile.open(game, p) for p in test_paths), key=lambda f: f.info.timestamp)

    if limit_file_count is not None:
        train_files = train_files[-min(limit_file_count, len(train_files)):]
        test_files = test_files[-min(limit_file_count, len(train_files)):]

    train_sampler = FileListSampler(FileList(game, train_files), batch_size, None, include_final, threads=1)
    test_sampler = FileListSampler(FileList(game, test_files), batch_size, None, include_final, threads=1)

    print(f"Train file count: {len(train_files)}")
    print(f"Train file game count: {sum(f.info.simulation_count for f in train_files)}")
    print(f"Train position count: {len(train_sampler)}")

    print(f"Test file count: {len(test_files)}")
    print(f"Train file game count: {sum(f.info.simulation_count for f in test_files)}")
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
        network = torch.jit.load(os.path.join(output_folder, f"network_{last_bi}.pt"))

    network.to(DEVICE)
    print_param_count(network)

    # optimizer = SGD(network.parameters(), weight_decay=1e-5, lr=0.0, momentum=0.9)
    # schedule = WarmupSchedule(100, FixedSchedule([0.02, 0.01, 0.001], [900, 2_000]))

    optimizer = torch.optim.AdamW(network.parameters(), weight_decay=1e-5)
    schedule = None

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
