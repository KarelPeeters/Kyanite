import glob
import os
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Adam

from lib.games import Game
from lib.model.lc0_pre_act import LCZOldPreNetwork
from lib.model.simple import SimpleNetwork
from lib.save_onnx import save_onnx


def build_networks():
    sttt = Game.find("sttt")
    networks = {
        "simple-sttt-1x64": (sttt, SimpleNetwork(sttt, 1, 64, False)),
        "simple-bn-sttt-1x64": (sttt, SimpleNetwork(sttt, 1, 64, True)),
        "pre-sttt-4x8": (sttt, LCZOldPreNetwork(sttt, 4, 8, 5, (6, 32))),
    }
    return networks


def save_networks(output_folder: str, check_batch_size: int, networks: Dict[str, Tuple[Game, nn.Module]]):
    for name, (game, network) in networks.items():
        # train for a few batches so all weight layers have meaningful values
        #   otherwise for example BN layers would start with lambda=1, bias=0
        print("Training network with random inputs")
        opt = Adam(network.parameters())
        for i in range(100):
            input = torch.randn(check_batch_size, *game.full_input_shape)
            outputs = network(input)
            total = sum(torch.sum(x.mean()) for x in outputs)

            opt.zero_grad()
            total.backward()
            opt.step()

        path = os.path.join(output_folder, f"{name}.onnx")
        save_onnx(game, path, network, check_batch_size)


def prepare_output_folder(output_folder):
    """ check whether the folder exists and only contains onnx and bin files, then delete those """
    assert os.path.exists(output_folder)
    for file in glob.glob(output_folder + "*"):
        assert file.endswith(".onnx") or file.endswith(".bin")
    for file in glob.glob(output_folder + "*"):
        os.remove(file)


CHECK_BATCH_SIZE = 2


def main():
    output_folder = "../../rust/cuda-nn-eval/tests/data/"

    prepare_output_folder(output_folder)
    networks = build_networks()
    save_networks(output_folder, CHECK_BATCH_SIZE, networks)


if __name__ == '__main__':
    main()
