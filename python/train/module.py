import torch.nn.functional as nnf
from torch import nn
from torch.optim import Adam

from data.games import Game
from data.view import GameDataView
from train.loss import cross_entropy_masked


class TrainModule:
    def __init__(self, game: Game, model: nn.Module):
        super().__init__()

        self.game = game
        self.model = model

    def training_step(self, batch, bi):
        view = GameDataView(self.game, batch)

        wdl_logit, policy_logit = self.model(view.input)

        wdl_loss_ce = cross_entropy_masked(wdl_logit, view.wdl_final, None)
        policy_loss_ce = cross_entropy_masked(policy_logit, view.policy, view.policy_mask)
        total_loss = wdl_loss_ce + policy_loss_ce

        # self.log("bi", bi)
        # self.log("total_loss", total_loss)
        # self.log("loss_wdl", wdl_loss_ce)
        # self.log("loss_policy", policy_loss_ce)

        wdl_loss_mse = nnf.mse_loss(nnf.softmax(wdl_logit, -1), view.wdl_final)
        # self.log("loss_wdl_mse", wdl_loss_mse)
        print(total_loss.item())

        return total_loss

    def configure_optimizers(self):
        return Adam(self.parameters())
