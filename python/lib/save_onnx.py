from pathlib import Path
from typing import Optional

import torch
from torch import nn

from lib.games import Game
from lib.util import guess_module_device


def save_onnx(game: Game, path_onnx: str, network: nn.Module, check_batch_size: Optional[int]):
    path_onnx = Path(path_onnx)
    assert path_onnx.suffix == ".onnx"
    print(f"Saving model to {path_onnx}")

    guessed_device = guess_module_device(network)
    network.eval()

    # calculate a real example
    used_batch_size = check_batch_size if check_batch_size is not None else 1
    check_input = torch.randn(used_batch_size, *game.full_input_shape)
    check_outputs = network(check_input.to(guessed_device))

    if check_batch_size is not None:
        # save the input and output for testing purposes
        path_check_bin = path_onnx.with_suffix(".bin")
        with open(path_check_bin, "wb") as f:
            f.write(check_batch_size.to_bytes(1, byteorder="little", signed=False))
            f.write(check_input.numpy().tobytes())
            for output in check_outputs:
                f.write(output.cpu().detach().numpy().tobytes())

    # move the network to cpu to get onnx exporting to work
    network.to("cpu")

    batch_axis = {0: "batch_size"}
    torch.onnx.export(
        model=network,
        args=check_input,
        f=path_onnx,
        example_outputs=check_outputs,
        input_names=["input"],
        output_names=["scalars", "policy"],
        dynamic_axes={"input": batch_axis, "scalars": batch_axis, "policy": batch_axis},
        opset_version=7,
    )

    # return the network to the original device
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
