import torch
from torch import nn

from util import Data


def evaluate_model(model, data: Data):
    # TODO try predicting win/loss/draw again
    #   maybe also use those to predict a value and add that as an 3rd term to the loss
    y_value_pred, y_move_pred = model(data)

    # mask out invalid moves and take the softmaxes
    y_move_pred = y_move_pred * data.mask
    y_move_pred = y_move_pred / y_move_pred.sum(-1, keepdim=True)

    # TODO try to get different loss functions to work
    loss = nn.functional.mse_loss(y_value_pred, data.y_value) + \
           nn.functional.mse_loss(y_move_pred, data.y_move_prob)

    value_acc = (y_value_pred.sign() == data.y_value.sign()).float().mean()
    move_prob_acc = (torch.argmax(y_move_pred, dim=1) == torch.argmax(data.y_move_prob, dim=1)).float().mean()

    return loss, value_acc, move_prob_acc


def train_model(model, optimizer, train_data, test_data, epochs: int):
    plot_legend = ["train_win_acc", "train_move_acc", "test_win_acc", "test_move_acc", "train_loss", "test_loss"]
    plot_data = torch.zeros(epochs, len(plot_legend))

    for epoch in range(epochs):
        model.train()
        train_loss, _, _ = evaluate_model(model, train_data)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        train_loss, train_win_acc, train_move_prob_acc = evaluate_model(model, train_data)
        test_loss, test_win_acc, test_move_prob_acc = evaluate_model(model, test_data)

        values = [train_win_acc, train_move_prob_acc, test_win_acc, test_move_prob_acc, train_loss, test_loss]
        plot_data[epoch, :] = torch.tensor(values)

        log = ", ".join((f"{name}={value:.2f}" for name, value in zip(plot_legend, values)))
        print(f"Epoch {epoch} {log}")

    return plot_data, plot_legend
