from dataclasses import dataclass

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


@dataclass
class TrainSettings:
    game: Game

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

        wdl_logit, policy_logit = network(view.input)
        wdl = nnf.softmax(wdl_logit, -1)

        # losses
        loss_wdl = nnf.mse_loss(wdl, view.wdl_final)
        loss_value = nnf.mse_loss(wdl[:, 0] - wdl[:, 2], view.wdl_final[:, 0] - view.wdl_final[:, 1])
        loss_policy = cross_entropy_masked(policy_logit, view.policy, view.policy_mask)
        loss_total = loss_wdl + self.policy_weight * loss_policy

        log("loss-wdl", f"{log_prefix} wdl", loss_wdl)
        log("loss-value", f"{log_prefix} value", loss_value)
        log("loss-policy", f"{log_prefix} policy", loss_policy)
        log("loss-total", f"{log_prefix} total", loss_total)

        # accuracies
        # TODO check that all of this calculates the correct values in the presence of pass moves
        # TODO actually, for games like ataxx just never ask the network about pass positions
        batch_size = len(batch)

        acc_wdl = (wdl_logit.argmax(dim=-1) == view.wdl_final.argmax(dim=-1)).sum() / batch_size

        policy_argmax = (policy_logit * view.policy_mask).flatten(1).argmax(dim=-1)
        acc_policy = (policy_argmax == view.policy.flatten(1).argmax(dim=-1)).sum() / batch_size
        acc_policy_captured = torch.gather(view.policy.flatten(1), 1, policy_argmax.view(-1, 1)).sum() / batch_size

        log("acc-wdl", f"{log_prefix} wdl", acc_wdl)
        log("acc-policy", f"{log_prefix} acc", acc_policy)
        log("acc-policy", f"{log_prefix} captured", acc_policy_captured)

        return loss_total


# TODO pin memory?
def batch_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    # noinspection PyTypeChecker
    random_sampler = RandomSampler(dataset, replacement=True)
    batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    return loader
