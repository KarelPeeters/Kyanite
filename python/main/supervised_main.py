import glob
import os
import re
from threading import Thread
from typing import Optional

import torch
from torch.optim import SGD

from lib.data.buffer import FileListSampler
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.model.post_act import PostActNetwork
from lib.plotter import LogPlotter, qt_app
from lib.schedule import FixedSchedule, WarmupSchedule
from lib.supervised import supervised_loop
from lib.train import TrainSettings, ValueTarget
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


def main():
    app = qt_app()

    train_pattern = f"../../data/pgn/*_large.json"
    test_pattern = f"../../data/pgn/*_large.json"
    output_folder = "../../data/supervised/lichess_09_2000/"

    game = Game.find("chess")
    os.makedirs(output_folder, exist_ok=True)

    batch_size = 1024

    test_steps = 16
    save_steps = 128

    settings = TrainSettings(
        game=game,
        value_target=ValueTarget.Zero,
        wdl_weight=0.5,
        value_weight=0.5,
        policy_weight=1.0,
        clip_norm=100,
        train_in_eval_mode=True,
    )

    # 200MB RAM for offsets
    max_positions = None

    train_files = [DataFile.open(game, path, max_positions) for path in glob.glob(train_pattern)]
    train_sampler = FileListSampler(game, train_files, batch_size)

    test_files = [DataFile.open(game, path, max_positions) for path in glob.glob(test_pattern)]
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
        network = PostActNetwork(game, 16, 128, 8, 128)
    else:
        logger = Logger.load(os.path.join(output_folder, "log.npz"))
        start_bi = last_bi + 1
        network = torch.jit.load(os.path.join(output_folder, f"network_{last_bi}.pb"))

    network.to(DEVICE)
    print_param_count(network)

    optimizer = SGD(network.parameters(), weight_decay=1e-5, lr=0.0, momentum=0.9)
    schedule = WarmupSchedule(100, FixedSchedule([0.02, 0.01, 0.001], [1_000, 16_000]))

    plotter = LogPlotter()
    plotter.update(logger)

    def thread_main():
        supervised_loop(
            settings, schedule, optimizer,
            start_bi, output_folder, logger, plotter,
            network, train_sampler, test_sampler,
            test_steps, save_steps,
        )

        # currently these never trigger (since the loop never stops), but that may change in the future
        train_sampler.close()
        test_sampler.close()

    thread = Thread(target=thread_main, daemon=True)
    thread.start()

    app.exec()


if __name__ == '__main__':
    main()
