import os.path
from argparse import ArgumentParser
from typing import Optional

import torch

from lib.games import Game
from lib.save_onnx import save_onnx


def main():
    parser = ArgumentParser()

    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--check_batch_size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)

    args = parser.parse_args()

    game: str = args.game
    check_batch_size: Optional[int] = args.check_batch_size
    force: bool = args.force
    input_path: str = args.input
    output_path: str = args.output

    assert input_path.endswith(".pt"), f"Input must be a .pt file"
    assert output_path.endswith(".onnx"), f"Output must be an .onnx file"

    assert os.path.exists(input_path), f"Input path does not exist: '{input_path}'"

    if os.path.exists(output_path):
        if not force and os.path.getmtime(output_path) > os.path.getmtime(input_path):
            print("Not converting network since .onnx output is newer than .pt input")
            return
        else:
            print("Overwriting existing .onnx output")
            os.remove(output_path)

    game: Game = Game.find(game)
    network = torch.jit.load(input_path)

    # bugfix for old networks that didn't register this attribute as a buffer
    if hasattr(network, "policy_head") and hasattr(network.policy_head, "FLAT_TO_ATT"):
        network.policy_head.FLAT_TO_ATT = network.policy_head.FLAT_TO_ATT.to("cpu")

    save_onnx(game, output_path, network, check_batch_size)


if __name__ == '__main__':
    main()
