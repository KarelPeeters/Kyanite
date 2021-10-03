from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from lib.data.buffer import FileBuffer
from lib.data.position import PositionBatch
from lib.games import Game
from lib.logger import Logger
from lib.util import calc_gradient_norms


@dataclass
class TrainSettings:
    game: Game

    wdl_weight: float
    value_weight: float
    policy_weight: float

    batch_size: int
    batches: int

    clip_norm: float

    def run_train(self, buffer: FileBuffer, optimizer: Optimizer, network: nn.Module, logger: Logger, scheduler=None):
        # noinspection PyTypeChecker
        visits_per_sample = self.batches * self.batch_size / len(buffer)
        logger.log_gen("small", "visits_per_sample", visits_per_sample)

        for bi in range(self.batches):
            batch = buffer.sample_batch(self.batch_size)

            logger.start_batch()

            optimizer.zero_grad(set_to_none=True)

            network.train()
            loss = self.evaluate_batch(network, "train", logger.log_batch, batch)
            loss.backward()

            grad_norm = clip_grad_norm_(network.parameters(), max_norm=self.clip_norm)
            optimizer.step()

            if scheduler is not None:
                logger.log_batch("schedule", "lr", scheduler.get_last_lr()[0])
                scheduler.step()

            grad_norms = calc_gradient_norms(network)
            logger.log_batch("grad_norm", "min", np.min(grad_norms))
            logger.log_batch("grad_norm", "mean", np.mean(grad_norms))
            logger.log_batch("grad_norm", "max", np.max(grad_norms))
            logger.log_batch("grad_norm", "torch", grad_norm)

            logger.finish_batch()

    def evaluate_batch(self, network: nn.Module, log_prefix: str, log, batch: PositionBatch):
        """Returns the total loss for the given batch while logging a bunch of statistics"""

        value_logit, wdl_logit, policy_logit = network(batch.input_full)

        value = torch.tanh(value_logit)
        wdl = nnf.softmax(wdl_logit, -1)

        batch_value = batch.value_final()

        # losses
        loss_wdl = nnf.mse_loss(wdl, batch.wdl_final)
        loss_value = nnf.mse_loss(value, batch_value)
        loss_policy, acc_policy, cap_policy = evaluate_policy(policy_logit, batch.policy_indices, batch.policy_values)
        loss_total = self.wdl_weight * loss_wdl + self.value_weight * loss_value + self.policy_weight * loss_policy

        log("loss-wdl", f"{log_prefix} wdl", loss_wdl)
        log("loss-value", f"{log_prefix} value", loss_value)
        log("loss-policy", f"{log_prefix} policy", loss_policy)
        log("loss-total", f"{log_prefix} total", loss_total)

        # accuracies
        # TODO check that all of this calculates the correct values in the presence of pass moves
        # TODO actually, for games like ataxx just never ask the network about pass positions
        batch_size = len(batch)

        acc_value = torch.eq(value.sign(), batch_value.sign()).sum() / (batch_value != 0).sum()
        acc_wdl = torch.eq(wdl_logit.argmax(dim=-1), batch.wdl_final.argmax(dim=-1)).sum() / batch_size

        log("acc-wdl", f"{log_prefix} value", acc_value)
        log("acc-wdl", f"{log_prefix} wdl", acc_wdl)
        log("acc-policy", f"{log_prefix} acc", acc_policy)
        log("acc-policy", f"{log_prefix} captured", cap_policy)

        return loss_total


def evaluate_policy(logits, indices, values):
    """Returns the cross-entropy loss, the accuracy and the value of the argmax policy."""
    assert len(indices.shape) == 2
    assert indices.shape == values.shape
    assert len(logits) == len(indices)
    (batch_size, max_mv_count) = indices.shape

    logits = logits.flatten(1)

    selected_logits = torch.gather(logits, 1, indices)

    selected_logits.cpu()[values.cpu() == -1] = -np.inf
    selected_logits[values == -1] = -np.inf

    loss = values * torch.log_softmax(selected_logits, 1)

    masked_moves = loss.isinf()
    loss[masked_moves] = 0
    total_loss = -loss.sum(axis=1).mean(axis=0)

    # accuracy
    selected_argmax = selected_logits.argmax(dim=1, keepdim=True)
    acc = torch.sum(torch.eq(selected_argmax, values.argmax(dim=1, keepdim=True))) / batch_size
    cap = torch.gather(values, 1, selected_argmax).mean()

    return total_loss, acc, cap