from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from lib.data.position import PositionBatch, UnrolledPositionBatch
from lib.games import Game
from lib.logger import Logger
from lib.networks import MuZeroNetworks
from lib.util import calc_gradient_norms, calc_parameter_norm, scale_gradient


class ScalarTarget:
    Final: 'ScalarTarget'
    Zero: 'ScalarTarget'

    def __init__(self, final: float):
        assert 0.0 <= final <= 1.0
        self.final = final

    def pick(self, final, zero):
        if self.final == 1.0:
            return final
        if self.final == 0.0:
            return zero

        return self.final * final + (1.0 - self.final) * zero


ScalarTarget.Final = ScalarTarget(1.0)
ScalarTarget.Zero = ScalarTarget(0.0)


@dataclass
class TrainSettings:
    game: Game
    scalar_target: ScalarTarget

    value_weight: float
    wdl_weight: float
    moves_left_weight: float
    moves_left_delta: float
    policy_weight: float

    train_in_eval_mode: bool
    clip_norm: float

    def train_step(self, batch: PositionBatch, network: nn.Module, optimizer: Optimizer, logger: Logger):
        optimizer.zero_grad(set_to_none=True)

        if self.train_in_eval_mode:
            network.eval()
        else:
            network.train()

        scalars, policy_logits = network(batch.input_full)
        loss = self.evaluate_batch_predictions("train", logger, batch, scalars, policy_logits)
        loss.backward()

        grad_norm = clip_grad_norm_(network.parameters(), max_norm=self.clip_norm)
        optimizer.step()

        grad_norms = calc_gradient_norms(network)
        logger.log("grad_norm", "min", np.min(grad_norms))
        logger.log("grad_norm", "mean", np.mean(grad_norms))
        logger.log("grad_norm", "max", np.max(grad_norms))
        logger.log("grad_norm", "torch", grad_norm)
        logger.log("param_norm", "param_norm", calc_parameter_norm(network))

    def train_step_unrolled(
            self,
            batch: UnrolledPositionBatch,
            networks: MuZeroNetworks, optimizer: Optimizer,
            logger: Logger
    ):
        optimizer.zero_grad(set_to_none=True)

        if self.train_in_eval_mode:
            networks.eval()
        else:
            networks.train()

        loss = self.evaluate_batch_unrolled(networks, "train", logger, batch)
        loss.backward()

        grad_norm = clip_grad_norm_(networks.parameters(), max_norm=self.clip_norm)
        optimizer.step()

        grad_norms = calc_gradient_norms(networks)
        logger.log("grad_norm", "min", np.min(grad_norms))
        logger.log("grad_norm", "mean", np.mean(grad_norms))
        logger.log("grad_norm", "max", np.max(grad_norms))
        logger.log("grad_norm", "torch", grad_norm)
        logger.log("param_norm", "param_norm", calc_parameter_norm(networks))

    def evaluate_batch_unrolled(
            self, networks: MuZeroNetworks,
            log_prefix: str, logger: Logger,
            batch: UnrolledPositionBatch,
    ):
        curr_state = networks.representation(batch.steps[0].input_full)
        scalars_0, policy_logits_0 = networks.prediction(curr_state)

        total_loss = self.evaluate_batch_predictions(f"{log_prefix}/f0", logger, batch.steps[0], scalars_0,
                                                     policy_logits_0)

        logger.log("state", f"max_0", torch.std(curr_state.flatten(1), dim=1).mean())

        for k in range(1, len(batch.steps)):
            curr_state = networks.dynamics(curr_state, batch.steps[k - 1].played_mv_full)
            scalars_k, policy_logits_k = networks.prediction(curr_state)

            total_loss += 1 / k * self.evaluate_batch_predictions(f"{log_prefix}/f{k}", logger, batch.steps[k],
                                                                  scalars_k, policy_logits_k)

            # TODO clamp/normalize curr_state somewhere
            logger.log("state", f"max_{k}", torch.std(curr_state.flatten(1), dim=1).mean())

            curr_state = scale_gradient(curr_state, 0.5)

        return total_loss

    def evaluate_batch_predictions(
            self,
            log_prefix: str, logger: Logger,
            batch: PositionBatch,
            scalars, policy_logits,
    ):
        """Returns the total loss for the given batch while logging a bunch of statistics"""

        value = torch.tanh(scalars[:, 0])
        wdl = nnf.softmax(scalars[:, 1:4], -1)
        moves_left = torch.relu(scalars[:, 4])

        batch_value = self.scalar_target.pick(final=batch.v_final, zero=batch.v_zero)
        batch_wdl = self.scalar_target.pick(final=batch.wdl_final, zero=batch.wdl_zero)
        # TODO this should be replaced with the same pick construct as the other ones
        batch_moves_left = batch.moves_left_final

        # losses
        loss_value = nnf.mse_loss(value, batch_value)
        loss_wdl = nnf.mse_loss(wdl, batch_wdl)
        loss_moves_left = nnf.huber_loss(moves_left, batch_moves_left, delta=self.moves_left_delta)
        loss_policy, acc_policy, cap_policy = evaluate_policy(policy_logits, batch.policy_indices, batch.policy_values)

        loss_total = self.combine_losses(log_prefix, logger, loss_value, loss_wdl, loss_moves_left, loss_policy)

        # accuracies
        # TODO check that all of this calculates the correct values in the presence of pass moves
        # TODO actually, for games like ataxx just never ask the network about pass positions
        batch_size = len(batch)

        acc_value = torch.eq(value.sign(), batch_value.sign()).sum() / (batch_value != 0).sum()
        acc_wdl = torch.eq(wdl.argmax(dim=-1), batch_wdl.argmax(dim=-1)).sum() / batch_size

        logger.log("acc-value", f"{log_prefix} value", acc_value)
        logger.log("acc-value", f"{log_prefix} wdl", acc_wdl)
        logger.log("acc-policy", f"{log_prefix} acc", acc_policy)
        logger.log("acc-policy", f"{log_prefix} captured", cap_policy)

        return loss_total

    def combine_losses(
            self, log_prefix: str, logger: Logger,
            value, wdl, moves_left, policy
    ):
        value_weighed = self.value_weight * value
        wdl_weighed = self.wdl_weight * wdl
        moves_left_weighed = self.moves_left_weight * moves_left
        policy_weighed = self.policy_weight * policy

        loss_total = value_weighed + wdl_weighed + moves_left_weighed + policy_weighed

        logger.log("loss-wdl", f"{log_prefix} wdl", wdl)
        logger.log("loss-value", f"{log_prefix} value", value)
        logger.log("loss-policy", f"{log_prefix} policy", policy)
        logger.log("loss-moves-left", f"{log_prefix} moves-left", moves_left)

        logger.log("loss-total", f"{log_prefix} total", loss_total)
        logger.log("loss-part", f"{log_prefix} value", value_weighed)
        logger.log("loss-part", f"{log_prefix} wdl", wdl_weighed)
        logger.log("loss-part", f"{log_prefix} moves_left", moves_left_weighed)
        logger.log("loss-part", f"{log_prefix} policy", policy_weighed)

        return loss_total


def evaluate_policy(logits, indices, values):
    """Returns the cross-entropy loss, the accuracy and the value of the argmax policy."""
    assert len(indices.shape) == 2
    assert indices.shape == values.shape
    assert len(logits) == len(indices)
    (batch_size, max_mv_count) = indices.shape

    if max_mv_count == 0:
        zero = torch.tensor(0.0).to(logits.device)
        one = torch.tensor(1.0).to(logits.device)
        return zero, one, one

    logits = logits.flatten(1)

    selected_logits = torch.gather(logits, 1, indices)
    selected_logits[values == -1] = -np.inf

    loss = values * torch.log_softmax(selected_logits, 1)

    # -inf for non-available moves, nan when there are no moves available at all
    masked_moves = ~loss.isfinite()
    loss[masked_moves] = 0
    total_loss = -loss.sum(axis=1).mean(axis=0)

    empty_count = (values == -1).all(axis=1).sum().float()

    # accuracy (top move matches) and captured (policy of net top move)
    selected_argmax = selected_logits.argmax(dim=1, keepdim=True)
    acc = torch.sum(torch.eq(selected_argmax, values.argmax(dim=1, keepdim=True))) / batch_size
    cap = torch.gather(values, 1, selected_argmax).mean() + 2 * (empty_count / batch_size)

    return total_loss, acc, cap
