import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf
from matplotlib import pyplot as plt
from torch.optim import SGD

from lib.data.file import DataFile
from lib.data.file_list import FileListSampler, FileList
from lib.games import Game
from lib.schedule import LinearSchedule
from lib.util import DEVICE

PIECE_NAMES = ["P", "N", "B", "R", "Q"]
POV_PIECE_COUNT = 5
PIECE_COUNT = 2 * POV_PIECE_COUNT

DATA_PATH = "../ignored/piece_values/data.npy"


def collect_training_data(data_gens, network_gens, batch_size: int, positions: int):
    game = Game.find("chess")

    files = [
        DataFile.open(game, f"C:/Documents/Programming/STTT/AlphaZero/data/loop/chess/16x128/selfplay/games_{gi}")
        for gi in data_gens
    ]
    sampler = FileListSampler(FileList(game, files), batch_size, None, False, 1)

    networks = []
    for gi in network_gens:
        path = f"C:/Documents/Programming/STTT/AlphaZero/data/loop/chess/16x128/training/gen_{gi}/network.pt"
        print(f"Loading {path}")

        network = torch.jit.load(path)
        network.eval()
        network.to(DEVICE)
        networks.append(network)

    data = np.zeros((positions, PIECE_COUNT + len(networks)))

    assert positions % batch_size == 0
    for bi in range(positions // batch_size):
        print(f"Data collection progress: {bi * batch_size} / {positions}")

        batch = sampler.next_batch()

        pieces_pov = batch.input_full[:, 8:13, :, :].sum(axis=(2, 3))
        pieces_opponent = batch.input_full[:, 14:19, :, :].sum(axis=(2, 3))

        data_slice = slice(bi * batch_size, (bi + 1) * batch_size)
        data[data_slice, 0:5] = pieces_pov.cpu().numpy()
        data[data_slice, 5:10] = pieces_opponent.cpu().numpy()

        with torch.no_grad():
            for ni, network in enumerate(networks):
                target_scalars, _ = network(batch.input_full)
                target_wdl = nnf.softmax(target_scalars[:, 1:4], -1)
                target_value = (target_wdl[:, 0] - target_wdl[:, 2])

                data[data_slice, PIECE_COUNT + ni] = target_value.cpu().numpy()

    sampler.close()

    print("Saving data")
    os.makedirs(Path(DATA_PATH).parent, exist_ok=True)
    np.save(DATA_PATH, data)


def fit_value_estimates(network_gens, batch_size: int, use_net_values: bool, bishop_pair: bool, knight_pair: bool,
                        rook_pair: bool):
    data = torch.tensor(np.load(DATA_PATH), dtype=torch.float32).to(DEVICE)

    if use_net_values:
        data_cols = [
            data[:, 0:5] - data[:, 5:10]
        ]
        legend = list(PIECE_NAMES)

        if bishop_pair:
            data_cols.append((data[:, 2, None] == 2).float() - (data[:, 7, None] == 2).float())
            legend.append("Bp")
        if knight_pair:
            data_cols.append((data[:, 1, None] == 2).float() - (data[:, 6, None] == 2).float())
            legend.append("Np")
        if rook_pair:
            data_cols.append((data[:, 3, None] == 2).float() - (data[:, 8, None] == 2).float())
            legend.append("Rp")

        input_data = torch.cat(data_cols, dim=1)
        y_lim = (-1, 10)
    else:
        assert not bishop_pair and not knight_pair, "Piece pairs not supported for non-net values"

        input_data = data[:, 0:10]
        legend = [f"+{n}" for n in PIECE_NAMES] + [f"-{n}" for n in PIECE_NAMES]
        y_lim = (-12, 12)

    input_size = input_data.shape[1]
    target_values = data[:, 10:]

    assert len(network_gens) == target_values.shape[1]
    final_values = np.zeros((len(network_gens), input_size))

    for ni, gi in enumerate(network_gens):
        linear_layer = torch.nn.Linear(input_size, 1)
        linear_layer.to(DEVICE)
        linear_layer.train()

        batches = 2000
        optimizer = SGD(linear_layer.parameters(), lr=0.0, momentum=0.1)
        schedule = LinearSchedule(0.2, 0.001, batches)

        history = np.zeros((batches, input_size))
        values_norm = np.nan

        for bi in range(batches):
            lr = schedule(bi)
            for group in optimizer.param_groups:
                group["lr"] = lr

            indices = torch.randint(len(data), (batch_size,), device=DEVICE)

            simple_value = torch.tanh(linear_layer(input_data[indices, :]))
            target_value = target_values[indices, ni, None]

            optimizer.zero_grad(True)
            loss = nnf.mse_loss(simple_value, target_value)
            loss.backward()
            optimizer.step()

            bias = linear_layer.bias.item()
            values_np = linear_layer.weight.cpu().detach().numpy()[0]
            values_norm = values_np / values_np[0]
            history[bi] = values_norm

            # print(f"ni: {ni} bi: {bi}, loss: {loss.item()}, bias: {bias}, values: {list(values_np)}, values_norm: {list(values_norm)}")

        final_values[ni, :] = values_norm
        print(f"ni={ni}, gi={gi}, values_norm={list(values_norm)}")

        plt.plot(history, label=legend)
        plt.title(f"Network {gi}")
        plt.xlabel("ei")
        plt.ylabel("piece value")
        plt.legend()
        plt.ylim(*y_lim)
        plt.show()

    plt.plot(network_gens, final_values, label=legend)
    plt.title("Piece value evolution during training")
    plt.xlabel("gi")
    plt.ylabel("piece value")
    plt.legend()
    plt.ylim(*y_lim)
    plt.show()


def main():
    network_gens = [15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 597, 700, 800,
                    900, 1000, 1250, 1500, 1750, 2000, 2400]
    network_gens += list(range(2500, 3700, 100))

    if False:
        collect_training_data(
            data_gens=range(2300, 2400),
            network_gens=network_gens,
            batch_size=1024,
            positions=1024 * 100,
        )

    fit_value_estimates(
        network_gens=network_gens,
        batch_size=128,
        use_net_values=True,
        bishop_pair=False,
        knight_pair=False,
        rook_pair=False,
    )


if __name__ == '__main__':
    main()
