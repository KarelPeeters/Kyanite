import torch
from torch import nn

from util import Data, DEVICE


def cross_entropy_masked(logits, target, mask):
    assert len(logits.shape) == 2
    assert logits.shape == target.shape

    log = torch.log_softmax(logits + mask.log(), dim=1)
    loss = -(target * log).nansum(dim=1)

    # average over batch dimension
    return loss.mean(dim=0)


def evaluate_model(model, data: Data):
    # TODO try predicting win/loss/draw again
    #   maybe also use those to predict a value and add that as an 3rd term to the loss
    y_value_pred, y_move_pred_logit = model(data.mask, data.x_tiles, data.x_macros)

    value_loss = nn.functional.mse_loss(y_value_pred, data.y_value)
    move_loss = cross_entropy_masked(y_move_pred_logit, data.y_move_prob, data.mask_flat)

    return value_loss, move_loss


def train_model(model, optimizer, train_data: Data, test_data: Data, epochs: int, train_batch_size: int,
                eval_batch_size: int):
    plot_legend = [
        "batch_value_loss", "batch_move_loss",
        "train_value_loss", "train_move_loss",
        "test_value_loss", "test_move_loss"
    ]
    plot_data = torch.zeros(epochs, len(plot_legend))

    for epoch in range(epochs):
        batch_count = len(train_data) // train_batch_size
        indices = torch.randperm(len(train_data), device=DEVICE)

        total_batch_value_loss = 0
        total_batch_move_loss = 0

        for bi in range(batch_count):
            batch_data = train_data.pick_batch(indices[bi * train_batch_size:(bi + 1) * train_batch_size])

            model.train()
            batch_value_loss, batch_move_loss = evaluate_model(model, batch_data)
            total_loss = batch_value_loss + batch_move_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_batch_value_loss += batch_value_loss.item()
            total_batch_move_loss += batch_move_loss.item()

        batch_value_loss = total_batch_value_loss / batch_count
        batch_move_loss = total_batch_move_loss / batch_count

        # TODO find the proper function for this randperm stuff
        model.eval()
        train_eval_batch = train_data.pick_batch(torch.randperm(len(train_data))[:eval_batch_size])
        train_value_loss, train_move_loss = evaluate_model(model, train_eval_batch)
        test_eval_batch = test_data.pick_batch(torch.randperm(len(test_data))[:eval_batch_size])
        test_value_loss, test_move_loss = evaluate_model(model, test_eval_batch)

        values = [
            batch_value_loss, batch_move_loss,
            train_value_loss, train_move_loss,
            test_value_loss, test_move_loss
        ]
        plot_data[epoch, :] = torch.tensor(values)

        log = ", ".join((f"{name}={value:.2f}" for name, value in zip(plot_legend, values)))
        print(f"Epoch {epoch} {log}")

    return plot_data, plot_legend
