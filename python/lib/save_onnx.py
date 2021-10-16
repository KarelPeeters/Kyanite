from pathlib import Path

import torch
from torch import nn

from lib.games import Game
from lib.util import guess_module_device


def save_onnx(game: Game, path_onnx: str, network: nn.Module, test_batch_size: int):
    path_onnx = Path(path_onnx)
    assert path_onnx.suffix == ".onnx"
    print(f"Saving model to {path_onnx}")

    guessed_device = guess_module_device(network)
    network.eval()

    # calculate a real example
    check_input = torch.randn(test_batch_size, *game.full_input_shape)
    check_outputs = network(check_input.to(guessed_device))

    # save the input and output for testing purposes
    path_check_bin = path_onnx.with_suffix(".bin")
    with open(path_check_bin, "wb") as f:
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
        output_names=["value", "wdl", "policy"],
        dynamic_axes={"input": batch_axis, "value": batch_axis, "wdl": batch_axis, "policy": batch_axis},
        opset_version=7,
    )

    # return the network to the original device
    network.to(guessed_device)
