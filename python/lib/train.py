from dataclasses import dataclass
from typing import Union

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
from lib.util import calc_gradient_norms, calc_parameter_norm, fake_quantize_scale, DEVICE


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

EitherNetwork = Union[nn.Module, MuZeroNetworks]
EitherBatch = Union[PositionBatch, UnrolledPositionBatch]


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

    mask_policy: bool

    def train_step(
            self,
            batch: EitherBatch,
            network: EitherNetwork,
            optimizer: Optimizer,
            logger: Logger
    ):
        if self.train_in_eval_mode:
            network.eval()
        else:
            network.train()

        loss = self.evaluate_either_batch(batch, network, logger, "train")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = clip_grad_norm_(network.parameters(), max_norm=self.clip_norm)
        optimizer.step()

        grad_norms = calc_gradient_norms(network)
        logger.log("grad_norm", "min", np.min(grad_norms))
        logger.log("grad_norm", "mean", np.mean(grad_norms))
        logger.log("grad_norm", "max", np.max(grad_norms))
        logger.log("grad_norm", "torch", grad_norm)
        logger.log("param_norm", "param_norm", calc_parameter_norm(network))

    def evaluate_either_batch(self, batch: EitherBatch, network: EitherNetwork, logger: Logger, log_prefix: str):
        if isinstance(batch, UnrolledPositionBatch):
            loss = self.evaluate_batch_unrolled(network, batch, log_prefix, logger)
        elif isinstance(batch, PositionBatch):
            loss = self.evaluate_batch(network, batch, log_prefix, logger)
        else:
            assert False, f"Unexpected batch type {type(batch)}"
        return loss

    def evaluate_batch(self, network: nn.Module, batch: PositionBatch, log_prefix: str, logger: Logger):
        scalars, policy_logits = network(batch.input_full)
        loss = self.evaluate_batch_predictions(log_prefix, logger, False, batch, scalars, policy_logits)
        return loss

    def evaluate_batch_unrolled(self, networks: MuZeroNetworks, batch: UnrolledPositionBatch, log_prefix: str,
                                logger: Logger):
        total_loss = torch.zeros((), device=DEVICE)
        curr_state = None

        for k, step in enumerate(batch.positions):
            if k == 0:
                curr_state = networks.representation(step.input_full)
            else:
                prev_position = batch.positions[k - 1]
                curr_state = networks.dynamics(curr_state, prev_position.played_mv_full)

            scalars_k, policy_logits_k = networks.prediction(curr_state)

            # limit the number of channels that have to be saved
            curr_state = curr_state[:, :networks.state_channels_saved, :, :]

            # quantize to reduce memory usage in inference, but only _after_ policy and value heads
            if networks.state_quant_bits is not None:
                curr_state = fake_quantize_scale(curr_state, 1.0, networks.state_quant_bits)

            total_loss += self.evaluate_batch_predictions(
                f"{log_prefix}/f{k}", logger, True,
                batch.positions[k], scalars_k, policy_logits_k
            )

            # TODO is a BN layer inside of the networks enough for hidden state normalization?
            std, mean = torch.std_mean(curr_state.flatten(1), dim=1)
            logger.log(f"state", f"{log_prefix} std_{k}", std.mean())
            logger.log(f"state", f"{log_prefix} mean_{k}", mean.mean())

            # TODO _why_ does the muzero paper scale the gradient here?
            #   maybe this is more important when working with SGD?
            # curr_state = scale_gradient(curr_state, 0.5)

        norm_loss = total_loss / len(batch.positions)
        return norm_loss

    def evaluate_batch_predictions(
            self,
            log_prefix: str, logger: Logger, log_policy_norm: bool,
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
        eval_policy = evaluate_policy(policy_logits, batch.policy_indices, batch.policy_values, self.mask_policy)

        loss_total = self.combine_losses(
            log_prefix, logger,
            loss_value, loss_wdl, loss_moves_left,
            eval_policy.train_loss
        )

        # value accuracies
        batch_size = len(batch)
        acc_value = torch.eq(value.sign(), batch_value.sign()).sum() / (batch_value != 0).sum()
        acc_wdl = torch.eq(wdl.argmax(dim=-1), batch_wdl.argmax(dim=-1)).sum() / batch_size

        logger.log("acc-value", f"{log_prefix} value", acc_value)
        logger.log("acc-value", f"{log_prefix} wdl", acc_wdl)

        # log policy info
        logger.log("acc-policy", f"{log_prefix} acc", eval_policy.norm_acc)
        logger.log("acc-policy", f"{log_prefix} top_mass", eval_policy.norm_top_mass)

        if not self.mask_policy:
            logger.log("acc-policy", f"{log_prefix} valid_mass", eval_policy.norm_valid_mass)

        if log_policy_norm:
            logger.log("loss-policy-norm", f"{log_prefix} policy", eval_policy.norm_loss)

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


def old_evaluate_policy(logits, indices, values):
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


@dataclass
class PolicyEvaluation:
    train_loss: torch.tensor
    norm_loss: torch.tensor

    norm_acc: torch.tensor
    norm_top_mass: torch.tensor
    norm_valid_mass: torch.tensor


VALUE_MASS_TOLERANCE = 0.01
LOG_CLIPPING = 10


def evaluate_policy(logits, indices, values, mask_invalid_moves: bool) -> PolicyEvaluation:
    """
    Returns the cross-entropy loss, the accuracy and the value of the argmax policy.
    The loss is calculated between `softmax(logits, 1)` and `torch.zeros(logits.shape).scatter(1, indices, values)`
    Indices 0 with value -1 are considered to be "unavailable" and not punished
    """
    assert len(indices.shape) == 2
    assert indices.shape == values.shape
    assert len(logits) == len(indices)

    (batch_size, max_mv_count) = indices.shape
    logits = logits.flatten(1)

    # for each batch element, whether there are any valid moves
    has_valid = (values != -1).any(dim=1)
    has_valid_count = has_valid.sum()

    # sum should be 1 or 0 (for no valid moves)
    value_mass = (values * (values != -1)).sum(dim=1)
    valid_and_1 = torch.logical_and(has_valid, (1.0 - value_mass < VALUE_MASS_TOLERANCE))
    invalid_and_0 = torch.logical_and(~has_valid, value_mass == 0.0)
    assert torch.logical_or(valid_and_1, invalid_and_0).all(), "Invalid value mass"

    # select data based on both indices and (values != -1)
    if mask_invalid_moves:
        # only softmax between selected indices, implicitly assuming other logits are -inf
        picked_logits = torch.gather(logits, 1, indices)
        picked_logits[values == -1] = -np.inf
        picked_logs = torch.log_softmax(picked_logits, 1).clamp(-LOG_CLIPPING, None)

        top_index = torch.argmax(picked_logits, dim=1)
        batch_acc = top_index == torch.argmax(values, dim=1)
        batch_top_mass = torch.gather(values, 1, top_index.unsqueeze(1)).squeeze(1) * has_valid
        batch_valid_mass = has_valid * np.nan
    else:
        # softmax between all logits, then only use selected logs, implicitly assuming other values are 0.0
        logs = torch.log_softmax(logits, 1).clamp(-LOG_CLIPPING, None)
        picked_logs = torch.gather(logs, 1, indices)
        picked_logs[values == -1] = -np.inf

        top_index = torch.argmax(logits, dim=1)
        batch_acc = top_index == torch.gather(indices, 1, torch.argmax(values, dim=1).unsqueeze(1)).squeeze(1)
        batch_top_mass = has_valid * np.nan

        predicted = torch.softmax(logits, 1)
        predicted_invalid = torch.scatter(predicted, 1, indices, (values == -1).float(), reduce="multiply")
        batch_valid_mass = 1 - predicted_invalid.sum(dim=1)

    # cross-entropy loss
    loss = -values * picked_logs

    # unavailable moves (values == -1 from before) become -inf but should be 0
    loss[loss.isinf()] = 0

    # positions without any valid moves become rows of nan, which we skip here
    total_loss = loss.nansum()
    return PolicyEvaluation(
        train_loss=total_loss / batch_size,
        norm_loss=total_loss / has_valid_count,
        norm_acc=(batch_acc * has_valid).sum() / has_valid_count,
        norm_top_mass=batch_top_mass.sum() / has_valid_count,
        norm_valid_mass=batch_valid_mass.sum() / has_valid_count,
    )
