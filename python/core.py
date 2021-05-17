import torch
from torch import nn

from util import Data


def evaluate_model(model, data: Data):
    # TODO try predicting win/loss/draw again
    #   maybe also use those to predict a value and add that as an 3rd term to the loss
    y_value_pred, y_move_pred = model(data)

    # mask out invalid moves and take the softmaxes (and renormalize)
    y_move_pred = y_move_pred * data.mask_flat
    y_move_pred = y_move_pred / y_move_pred.sum(-1, keepdim=True)

    # TODO try to get different loss functions to work
    loss = nn.functional.mse_loss(y_value_pred, data.y_value) + \
           nn.functional.mse_loss(y_move_pred, data.y_move_prob)

    value_acc = (y_value_pred.sign() == data.y_value.sign()).float().mean()
    move_prob_acc = (torch.argmax(y_move_pred, dim=1) == torch.argmax(data.y_move_prob, dim=1)).float().mean()

    return loss, value_acc, move_prob_acc


def train_model(model, optimizer, train_data: Data, test_data: Data, epochs: int, train_batch_size: int,
                eval_batch_size: int):
    plot_legend = ["train_win_acc", "train_move_acc", "test_win_acc", "test_move_acc", "train_loss", "test_loss"]
    plot_data = torch.zeros(epochs, len(plot_legend))

    for epoch in range(epochs):
        batch_count = len(train_data) // train_batch_size
        indices = torch.randperm(len(train_data))

        for bi in range(batch_count):
            batch_data = train_data.pick_batch(indices[bi * train_batch_size:(bi + 1) * train_batch_size])

            model.train()
            train_loss, _, _ = evaluate_model(model, batch_data)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # TODO find the proper function for this randperm stuff
        model.eval()
        train_eval_batch = train_data.pick_batch(torch.randperm(len(train_data))[:eval_batch_size])
        train_loss, train_win_acc, train_move_prob_acc = evaluate_model(model, train_eval_batch)
        test_eval_batch = test_data.pick_batch(torch.randperm(len(test_data))[:eval_batch_size])
        test_loss, test_win_acc, test_move_prob_acc = evaluate_model(model, test_eval_batch)

        values = [train_win_acc, train_move_prob_acc, test_win_acc, test_move_prob_acc, train_loss, test_loss]
        plot_data[epoch, :] = torch.tensor(values)

        log = ", ".join((f"{name}={value:.2f}" for name, value in zip(plot_legend, values)))
        print(f"Epoch {epoch} {log}")

    return plot_data, plot_legend
