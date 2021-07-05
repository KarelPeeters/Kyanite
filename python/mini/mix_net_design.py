import torch
from torch.onnx import TrainingMode

from models import MixModel

message_size = 5
batch_size = 2

model = MixModel(depth=4, message_size=16, wdl_size=32)

input_board = torch.randn(batch_size, 2, 10, 9)
output_wdl, output_policy = model(input_board)

print(output_wdl.shape)
print(output_policy.shape)

jit_model = torch.jit.script(model)
torch.jit.save(jit_model, "../../data/mix/mix_model.pt")

torch.onnx.export(
    model,
    args=input_board,
    f="../../data/mix/mix_model.onnx",
    example_outputs=(output_wdl, output_policy),
    opset_version=12,
    input_names=["input_board"],
    output_names=["output_wdl", "output_policy"],
    dynamic_axes={"input_board": {0: "batch_size"}, "output_wdl": {0: "batch_size"},
                  "output_policy": {0: "batch_size"}},
    training=TrainingMode.EVAL,
)
