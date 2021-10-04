import torch
from torch import nn

from lib.games import Game
from lib.util import DEVICE


def save_onnx(game: Game, path: str, network: nn.Module):
    assert path.endswith(".onnx")
    print(f"Saving model to {path}")

    network.eval()
    network.to("cpu")
    example_input = torch.zeros(1, *game.full_input_shape)
    example_outputs = network(example_input)

    batch_axis = {0: "batch_size"}
    torch.onnx.export(
        model=network,
        args=example_input,
        f=path,
        example_outputs=example_outputs,
        input_names=["input"],
        output_names=["value", "wdl", "policy"],
        dynamic_axes={"input": batch_axis, "value": batch_axis, "wdl": batch_axis, "policy": batch_axis},
        opset_version=7,
    )
    network.to(DEVICE)
