import torch
from torch import nn

from lib.games import Game
from lib.util import DEVICE


def save_onnx(game: Game, path: str, network: nn.Module):
    assert path.endswith(".onnx")
    print(f"Saving model to {path}")

    network.eval()
    network.to("cpu")
    example_input = torch.zeros(1, *game.input_shape_history)
    example_outputs = network(example_input)
    torch.onnx.export(
        model=network,
        args=example_input,
        f=path,
        example_outputs=example_outputs,
        input_names=["input"],
        output_names=["wdl", "policy"],
        dynamic_axes={"input": {0: "batch_size"}, "wdl": {0: "batch_size"}, "policy": {0: "batch_size"}},
    )
    network.to(DEVICE)
