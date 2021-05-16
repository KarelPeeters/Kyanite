import torch

from util import Data


def evaluate_model(model, loss_func, data: Data):
    y_pred = model(data)
    loss = loss_func(y_pred, data.y_win)
    acc = (torch.argmax(y_pred, dim=1) == torch.argmax(data.y_win, dim=1)).float().mean()

    return loss, acc


def train_model(model, optimizer, loss_func, train_data, test_data, epochs: int):
    plot_data = torch.zeros(epochs, 4)
    plot_legend = ["train_loss", "train_acc", "test_loss", "test_acc"]

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = evaluate_model(model, loss_func, train_data)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        train_loss, train_acc = evaluate_model(model, loss_func, train_data)
        test_loss, test_acc = evaluate_model(model, loss_func, test_data)

        values = [train_loss, train_acc, test_loss, test_acc]
        plot_data[epoch, :] = torch.tensor(values)

        log = ", ".join((f"{name}={value:.2f}" for name, value in zip(plot_legend, values)))
        print(f"Epoch {epoch} {log}")

    return plot_data, plot_legend
