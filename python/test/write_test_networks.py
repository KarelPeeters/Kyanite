import glob
import os
from typing import Dict, Tuple

from torch import nn

from lib.games import Game
from lib.model.lc0_pre_act import LCZOldPreNetwork
from lib.model.simple import SimpleNetwork
from lib.save_onnx import save_onnx


def build_networks():
    chess = Game.find("chess")
    sttt = Game.find("sttt")
    networks = {
        "simple-sttt-1x64": (sttt, SimpleNetwork(sttt, 1, 64)),
        "pre-sttt-4x8": (sttt, LCZOldPreNetwork(sttt, 4, 8, 5, (6, 32))),
        # "lcz_old_pre_0x1": (chess, LCZOldPreNetwork(chess, 0, 1)),
        # "lcz_old_pre_4x8": (chess, LCZOldPreNetwork(chess, 4, 8)),
    }
    return networks


def save_networks(output_folder: str, check_batch_size: int, networks: Dict[str, Tuple[Game, nn.Module]]):
    for name, (game, network) in networks.items():
        path = os.path.join(output_folder, f"{name}.onnx")
        save_onnx(game, path, network, check_batch_size)


def prepare_output_folder(output_folder):
    """ check whether the folder exists and only contains onnx and bin files, then delete those """
    assert os.path.exists(output_folder)
    for file in glob.glob(output_folder + "*"):
        assert file.endswith(".onnx") or file.endswith(".bin")
    for file in glob.glob(output_folder + "*"):
        os.remove(file)


def main():
    output_folder = "../../rust/cuda-nn-eval/tests/data/"
    check_batch_size = 2

    prepare_output_folder(output_folder)
    networks = build_networks()
    save_networks(output_folder, check_batch_size, networks)


if __name__ == '__main__':
    main()
