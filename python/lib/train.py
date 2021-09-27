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

    def run_train(self, dataset: Dataset, optimizer: Optimizer, network: nn.Module, logger: Logger, scheduler=None):
        loader = batch_loader(dataset, self.batch_size)

        # noinspection PyTypeChecker
        visits_per_sample = self.batches * self.batch_size / len(dataset)
        logger.log_gen("small", "visits_per_sample", visits_per_sample)

        for bi, batch in enumerate(loader):
            if bi >= self.batches:
                break
            batch = batch.to(DEVICE)

            logger.start_batch()

            optimizer.zero_grad(set_to_none=True)

            network.train()
            loss = self.evaluate_loss(network, "train", logger.log_batch, batch)
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

    def evaluate_loss(self, network: nn.Module, log_prefix: str, log, batch: torch.Tensor):
        view = GameDataView(self.game, batch, includes_history=True)

        torch.set_printoptions(threshold=np.inf)
        f = open("test.txt", "w")
        print(view.input[0, :, :, :], file=f)
        print(view.wdl_final[0, :], file=f)

        value_logit, policy_logit = network(view.input)
        value = torch.tanh(value_logit).squeeze(1) * 1.01

        value_final = view.wdl_final[:, 0] - view.wdl_final[:, 1]
        value_est = view.wdl_est[:, 0] - view.wdl_est[:, 1]

        loss_wdl_final_mse = nnf.mse_loss(value, value_final)
        loss_wdl_est_mse = nnf.mse_loss(value, value_est)

        loss_policy_ce = cross_entropy_masked(policy_logit, view.policy, view.policy_mask)

        loss_wdl = self.wdl_target.select(loss_wdl_final_mse, loss_wdl_est_mse)
        loss_total = loss_wdl + self.policy_weight * loss_policy_ce

        log("loss-wdl", f"{log_prefix} est_mse", loss_wdl_est_mse)
        log("loss-wdl", f"{log_prefix} final_mse", loss_wdl_final_mse)
        log("loss-policy", f"{log_prefix} policy_ce", loss_policy_ce)
        log("loss-total", f"{log_prefix} total", loss_total)

        batch_size = len(batch)
        value_final_acc = (value.sign() == value_final.sign()).sum() / batch_size
        value_est_acc = (value.sign() == value_est.sign()).sum() / batch_size

        policy_acc = (
                             torch.argmax(policy_logit.view(batch_size, -1), dim=1) ==
                             torch.argmax(view.policy.view(batch_size, -1), dim=1)
                     ).sum() / batch_size

        log("accuracy", f"{log_prefix} value final", value_final_acc)
        log("accuracy", f"{log_prefix} value est", value_est_acc)
        log("accuracy", f"{log_prefix} policy", policy_acc)

        return loss_total


# TODO pin memory?
def batch_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    # noinspection PyTypeChecker
    random_sampler = RandomSampler(dataset, replacement=True)
    batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    return loader
