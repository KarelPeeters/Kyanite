import json
import os.path
import warnings
from pathlib import Path
from typing import Optional, List

import torch
from torch import nn

from lib.games import Game
from lib.networks import MuZeroNetworks
from lib.util import guess_module_device


def save_muzero_onnx(game: Game, path_base: str, networks: MuZeroNetworks, check_batch_size: Optional[int]):
    assert path_base.endswith("_"), f"Path must end with '_', got '{path_base}'"

    state_shape = (networks.state_channels, game.board_size, game.board_size)
    state_limit_shape = (networks.state_channels_saved, game.board_size, game.board_size)

    info_path = path_base + "info.json"
    assert not os.path.exists(info_path), f"Path '{info_path}' already exists"

    info = {
        "game": game.name,
        "state_channels": networks.state_channels,
        "state_channels_saved": networks.state_channels_saved,
        "state_quant_bits": networks.state_quant_bits,
    }
    with open(info_path, "w") as f:
        json.dump(info, f)

    save_onnx_inner(
        path_base + "representation.onnx",
        networks.representation,
        [game.full_input_shape],
        ["input"], ["state"],
        check_batch_size
    )

    save_onnx_inner(
        path_base + "dynamics.onnx",
        networks.dynamics,
        [state_limit_shape, game.input_mv_shape],
        ["prev_state", "move"],
        ["state"],
        check_batch_size
    )

    save_onnx_inner(
        path_base + "prediction.onnx",
        networks.prediction,
        [state_shape],
        ["state"],
        ["scalars", "policy"],
        check_batch_size
    )


def save_onnx(game: Game, path_onnx: str, network: nn.Module, check_batch_size: Optional[int]):
    save_onnx_inner(
        path_onnx,
        network, [game.full_input_shape],
        ["input"], ["scalars", "policy"],
        check_batch_size
    )


def save_onnx_inner(
        path_onnx,
        network: nn.Module, input_shapes,
        input_names: List[str], output_names: List[str],
        check_batch_size: Optional[int]
):
    path_onnx = Path(path_onnx)
    assert path_onnx.suffix == ".onnx", f"Output path should end with .onnx: '{path_onnx}'"
    assert not path_onnx.exists(), f"Output path already exists: '{path_onnx}'"

    print(f"Saving model to {path_onnx}")

    guessed_device = guess_module_device(network)
    network.eval()

    # calculate a real example
    used_batch_size = check_batch_size if check_batch_size is not None else 1
    check_inputs = [
        torch.randn(used_batch_size, *shape)
        for shape in input_shapes
    ]

    check_inputs_device = [x.to(guessed_device) for x in check_inputs]
    check_outputs_device = network(*check_inputs_device)

    # optionally save the inputs and outputs for testing purposes
    if check_batch_size is not None:
        path_check_bin = path_onnx.with_suffix(".bin")
        with open(path_check_bin, "wb") as f:
            f.write(check_batch_size.to_bytes(1, byteorder="little", signed=False))
            for input in check_inputs:
                f.write(input.numpy().tobytes())
            for output in check_outputs_device:
                f.write(output.cpu().detach().numpy().tobytes())

    # move the network to cpu to get onnx exporting to work
    network.to("cpu")

    batch_axis = {0: "batch_size"}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model has no forward function")
        torch.onnx.export(
            model=network,
            args=tuple(check_inputs),
            f=path_onnx,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={k: batch_axis for k in input_names + output_names},
            opset_version=10,
        )

    # move the network back to the original device
    network.to(guessed_device)


# Based on https://github.com/microsoft/onnxruntime/blob/master/tools/python/remove_initializer_from_input.py
def remove_initializers_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
