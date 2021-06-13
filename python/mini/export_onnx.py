import os

import torch.jit
from torch.onnx import TrainingMode

from models import GoogleModel

model = GoogleModel(channels=64, blocks=4, value_channels=4, value_size=32, policy_channels=8, res=True)
# model = torch.jit.script(model)

model.to("cpu")
model.eval()

BATCH_SIZE = 100

batch_size = 17
input = torch.zeros(BATCH_SIZE, 5, 9, 9)
output = model(input)

os.makedirs("../../data/onnx", exist_ok=True)

torch.onnx.export(
    model,
    input,
    "../../data/onnx/train.onnx",
    example_outputs=output,
    opset_version=13,
    input_names=["input"],
    output_names=["value", "policy"],
    dynamic_axes={"input": {0: "batch_size"}, "policy": {0: "batch_size"}},
    training=TrainingMode.TRAINING,
    do_constant_folding=False,
)

torch.onnx.export(
    model,
    input,
    f"../../data/onnx/train_{BATCH_SIZE}.onnx",
    example_outputs=output,
    opset_version=13,
    input_names=["input"],
    output_names=["value", "policy"],
    # dynamic_axes={"input": {0: "batch_size"}, "policy": {0: "batch_size"}},
    training=TrainingMode.TRAINING,
    do_constant_folding=False,
)

torch.onnx.export(
    model,
    input,
    "../../data/onnx/eval.onnx",
    example_outputs=output,
    opset_version=13,
    input_names=["input"],
    output_names=["value", "policy"],
    dynamic_axes={"input": {0: "batch_size"}, "policy": {0: "batch_size"}},
    training=TrainingMode.EVAL,
)
