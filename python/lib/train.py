from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, RandomSampler, DataLoader, Dataset

from lib.dataview import GameDataView
from lib.games import Game
from lib.logger import Logger
from lib.loss import cross_entropy_masked
from lib.util import DEVICE, calc_gradient_norms


class WdlTarget(Enum):
    Final = auto()
    Estimate = auto()

    def select(self, final, est):
        if self == WdlTarget.Final:
            return final
        if self == WdlTarget.Estimate:
            return est
        assert False, self


class WdlLoss(Enum):
    CrossEntropy = auto()
    MSE = auto()

    def select(self, ce, mse):
        if self == WdlLoss.CrossEntropy:
            return ce
        if self == WdlLoss.MSE:
            return mse
        assert False, self


@dataclass
class TrainSettings:
    game: Game

    wdl_target: WdlTarget
    wdl_loss: WdlLoss
    policy_weight: float

    batch_size: int
    batches: int

    clip_norm: float

    def run_train(self, dataset: Dataset, optimizer: Optimizer, network: nn.Module, logger: Logger):
        loader = batch_loader(dataset, self.batch_size)

        # noinspection PyTypeChecker
        visits_per_sample = self.batches * self.batch_size / len(dataset)
        logger.log_gen("small", "visits_per_sample", visits_per_sample)

        for bi, batch in enumerate(loader):
            if bi > self.batches:
                break
            batch = batch.to(DEVICE)

            logger.start_batch()

            optimizer.zero_grad(set_to_none=True)

            network.train()
            loss = self.evaluate_loss(network, "train", logger.log_batch, batch)
            loss.backward()

            grad_norm = clip_grad_norm_(network.parameters(), max_norm=self.clip_norm)
            optimizer.step()

            grad_norms = calc_gradient_norms(network)
            logger.log_batch("grad_norm", "min", np.min(grad_norms))
            logger.log_batch("grad_norm", "mean", np.mean(grad_norms))
            logger.log_batch("grad_norm", "max", np.max(grad_norms))
            logger.log_batch("grad_norm", "torch", grad_norm)

            logger.finish_batch()

    def evaluate_loss(self, network: nn.Module, log_prefix: str, log, batch: torch.Tensor):
        view = GameDataView(self.game, batch)

        wdl_logit, policy_logit = network(view.input)

        loss_wdl_final_ce = cross_entropy_masked(wdl_logit, view.wdl_final, None)
        loss_wdl_final_mse = nnf.mse_loss(nnf.softmax(wdl_logit, -1), view.wdl_final)
        loss_wdl_est_ce = cross_entropy_masked(wdl_logit, view.wdl_est, None)
        loss_wdl_est_mse = nnf.mse_loss(nnf.softmax(wdl_logit, -1), view.wdl_est)

        loss_policy_ce = cross_entropy_masked(policy_logit, view.policy, view.policy_mask)

        loss_wdl = self.wdl_target.select(
            final=self.wdl_loss.select(ce=loss_wdl_final_ce, mse=loss_wdl_final_mse),
            est=self.wdl_loss.select(ce=loss_wdl_est_ce, mse=loss_wdl_est_mse),
        )
        loss_total = loss_wdl + self.policy_weight * loss_policy_ce

        if False and log_prefix == "train":
            log("loss-wdl", f"{log_prefix} final_ce", loss_wdl_final_ce)
            log("loss-wdl", f"{log_prefix} est_ce", loss_wdl_est_ce)
            log("loss-wdl", f"{log_prefix} est_mse", loss_wdl_est_mse)

        log("loss-wdl", f"{log_prefix} final_mse", loss_wdl_final_mse)
        log("loss-policy", f"{log_prefix} policy_ce", loss_policy_ce)
        log("loss-total", f"{log_prefix} total", loss_total)

        wdl_argmax = torch.argmax(wdl_logit, dim=1)
        final_argmax = torch.argmax(view.wdl_final, dim=1)
        est_argmax = torch.argmax(view.wdl_est, dim=1)

        batch_size = len(batch)
        wdl_final_acc = (wdl_argmax == final_argmax).sum() / batch_size
        wdl_est_acc = (wdl_argmax == est_argmax).sum() / batch_size
        wdl_cross_acc = (final_argmax == est_argmax).sum() / batch_size

        policy_acc = (
                             torch.argmax(policy_logit.view(batch_size, -1), dim=1) ==
                             torch.argmax(view.policy.view(batch_size, -1), dim=1)
                     ).sum() / batch_size

        log("accuracy", f"{log_prefix} wdl final", wdl_final_acc)
        log("accuracy", f"{log_prefix} wdl est", wdl_est_acc)
        log("accuracy", f"{log_prefix} wdl final vs cross", wdl_cross_acc)
        log("accuracy", f"{log_prefix} policy", policy_acc)

        return loss_total


# TODO pin memory?
def batch_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    # noinspection PyTypeChecker
    random_sampler = RandomSampler(dataset, replacement=True)
    batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    return loader
