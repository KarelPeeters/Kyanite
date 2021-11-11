from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from lib.data.buffer import FileListSampler
from lib.data.position import PositionBatch
from lib.games import Game
from lib.logger import Logger
from lib.util import calc_gradient_norms


class ValueTarget(Enum):
    Final = auto()
    Zero = auto()

    def pick(self, final, zero):
        if self == ValueTarget.Final:
            return final
        if self == ValueTarget.Zero:
            return zero
        assert False, self


@dataclass
class TrainSettings:
    game: Game
    value_target: ValueTarget

    wdl_weight: float
    value_weight: float
    policy_weight: float

    train_in_eval_mode: bool

    clip_norm: float

    def train_step(self, sampler: FileListSampler, network: nn.Module, optimizer: Optimizer, logger: Logger):
        batch = sampler.next_batch()

        optimizer.zero_grad(set_to_none=True)

        if self.train_in_eval_mode:
            network.eval()
        else:
            network.train()

        loss = self.evaluate_batch(network, "train", logger, batch, self.value_target)
        loss.backward()

        grad_norm = clip_grad_norm_(network.parameters(), max_norm=self.clip_norm)
        optimizer.step()

        grad_norms = calc_gradient_norms(network)
        logger.log("grad_norm", "min", np.min(grad_norms))
        logger.log("grad_norm", "mean", np.mean(grad_norms))
        logger.log("grad_norm", "max", np.max(grad_norms))
        logger.log("grad_norm", "torch", grad_norm)

        param_norm = sum(param.detach().norm(p=2) for param in network.parameters()).item()
        logger.log("param_norm", "param_norm", param_norm)

    def evaluate_batch(
            self, network: nn.Module,
            log_prefix: str, logger: Logger,
            batch: PositionBatch,
            value_target: ValueTarget
    ):
        """Returns the total loss for the given batch while logging a bunch of statistics"""

        value_logit, wdl_logit, policy_logit = network(batch.input_full)

        value = torch.tanh(value_logit)
        wdl = nnf.softmax(wdl_logit, -1)

        batch_value = value_target.pick(final=batch.value_final(), zero=batch.value_zero())
        batch_wdl = value_target.pick(final=batch.wdl_final, zero=batch.wdl_zero)

        # losses
        loss_wdl = nnf.mse_loss(wdl, batch_wdl)
        loss_value = nnf.mse_loss(value, batch_value)
        loss_policy, acc_policy, cap_policy = evaluate_policy(policy_logit, batch.policy_indices, batch.policy_values)
        loss_total = self.wdl_weight * loss_wdl + self.value_weight * loss_value + self.policy_weight * loss_policy

        logger.log("loss-wdl", f"{log_prefix} wdl", loss_wdl)
        logger.log("loss-value", f"{log_prefix} value", loss_value)
        logger.log("loss-policy", f"{log_prefix} policy", loss_policy)
        logger.log("loss-total", f"{log_prefix} total", loss_total)

        # accuracies
        # TODO check that all of this calculates the correct values in the presence of pass moves
        # TODO actually, for games like ataxx just never ask the network about pass positions
        batch_size = len(batch)

        acc_value = torch.eq(value.sign(), batch_value.sign()).sum() / (batch_value != 0).sum()
        acc_wdl = torch.eq(wdl_logit.argmax(dim=-1), batch_wdl.argmax(dim=-1)).sum() / batch_size

        logger.log("acc-value", f"{log_prefix} value", acc_value)
        logger.log("acc-value", f"{log_prefix} wdl", acc_wdl)
        logger.log("acc-policy", f"{log_prefix} acc", acc_policy)
        logger.log("acc-policy", f"{log_prefix} captured", cap_policy)

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
