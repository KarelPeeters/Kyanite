from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
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

    def run_train(self, dataset: Dataset, optimizer: Optimizer, network: nn.Module, logger: Logger):
        # noinspection PyTypeChecker
        random_sampler = RandomSampler(dataset, replacement=True)
        batch_sampler = BatchSampler(random_sampler, batch_size=self.batch_size, drop_last=True)
        loader = DataLoader(dataset, batch_sampler=batch_sampler)

        # noinspection PyTypeChecker
        visits_per_sample = self.batches * self.batch_size / len(dataset)
        logger.log_gen("small", "visits_per_sample", visits_per_sample)

        for batch in loader:
            batch = batch.to(DEVICE)

            logger.start_batch()

            optimizer.zero_grad(set_to_none=True)
            loss = self.evaluate_loss(network, logger, batch)
            loss.backward()
            optimizer.step()

            grad_norms = calc_gradient_norms(network)
            logger.log_batch("grad_norm", "min", np.min(grad_norms))
            logger.log_batch("grad_norm", "mean", np.mean(grad_norms))
            logger.log_batch("grad_norm", "max", np.max(grad_norms))

            logger.finish_batch()

    def evaluate_loss(self, network: nn.Module, logger: Logger, batch: torch.Tensor):
        network.train()
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

        logger.log_batch("loss", "wdl_final_ce", loss_wdl_final_ce)
        logger.log_batch("loss", "wdl_final_mse", loss_wdl_final_mse)
        logger.log_batch("loss", "wdl_est_ce", loss_wdl_est_ce)
        logger.log_batch("loss", "wdl_est_mse", loss_wdl_est_mse)
        logger.log_batch("loss", "policy_ce", loss_policy_ce)
        logger.log_batch("loss", "total", loss_total)

        return loss_total
