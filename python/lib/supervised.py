import itertools
import json
import os
import time
from typing import Optional

import torch
import torch.nn.functional as nnf
from torch import nn
from torch.optim import Optimizer

from lib.data.sampler import PositionSampler
from lib.logger import Logger
from lib.plotter import LogPlotter
from lib.save_onnx import save_onnx
from lib.schedule import Schedule
from lib.train import TrainSettings
from lib.util import json_map


def supervised_loop(
        settings: TrainSettings, schedule: Optional[Schedule], optimizer: Optimizer,
        start_bi: int, output_folder: str,
        logger: Logger, plotter: LogPlotter,
        network: nn.Module,
        train_sampler: PositionSampler, test_sampler: PositionSampler,
        test_steps: int, save_steps: int,
):
    with open(os.path.join(output_folder, f"settings_{start_bi}.json"), "w") as settings_f:
        json.dump(settings, settings_f, default=json_map, indent=2)

    prev_start = time.perf_counter()

    for bi in itertools.count(start_bi):
        plotter.block_while_paused()
        logger.start_batch()

        start = time.perf_counter()
        throughput = 1 / (start - prev_start)
        prev_start = start

        print(f"Starting batch {bi}")
        if bi != 0:
            print(f"Throughput: {throughput:.2f} batches/s")

        if schedule is not None:
            lr = schedule(bi)
            logger.log("schedule", "lr", lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

        train_batch = train_sampler.next_batch()
        settings.train_step(train_batch, network, optimizer, logger)

        if bi % test_steps == 0:
            network.eval()

            train_batch = train_sampler.next_batch()
            settings.evaluate_batch(network, train_batch, "test-train", logger)

            test_batch = test_sampler.next_batch()
            settings.evaluate_batch(network, test_batch, "test-test", logger)

            # compare to just predicting the mean value
            train_batch_wdl = settings.scalar_target.pick(final=train_batch.wdl_final, zero=train_batch.wdl_zero)
            test_batch_wdl = settings.scalar_target.pick(final=test_batch.wdl_final, zero=test_batch.wdl_zero)

            mean_wdl = train_batch_wdl.mean(axis=0, keepdims=True)
            mean_value = mean_wdl[0, 0] - mean_wdl[0, 2]

            loss_wdl_mean = nnf.cross_entropy(mean_wdl.expand(len(test_batch), 3), test_batch_wdl)

            test_batch_value = test_batch_wdl[:, 0] - test_batch_wdl[:, 2]
            loss_value_mean = nnf.mse_loss(mean_value.expand(len(test_batch)), test_batch_value)

            logger.log("loss-wdl", "trivial", loss_wdl_mean.item())
            logger.log("loss-value", "trivial", loss_value_mean.item())

            plotter.update(logger)

        if bi % save_steps == 0:
            print("Saving log")
            logger.save(os.path.join(output_folder, "log.npz"))

            print("Saving network")
            torch.jit.script(network).save(os.path.join(output_folder, f"network_{bi}.pt"))
            save_onnx(settings.game, os.path.join(output_folder, f"network_{bi}.onnx"), network, 4)
