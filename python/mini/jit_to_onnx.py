import os

import torch.jit
from torch.onnx import TrainingMode

folder = "../../data/esat2/modest/"
device = "cpu"

for file_pt in os.listdir(folder):
    if not file_pt.endswith(".pt"):
        continue
    file_onnx = os.path.splitext(file_pt)[0] + ".onnx"
    path_pt = os.path.join(folder, file_pt)
    path_onnx = os.path.join(folder, file_onnx)

    print(f"Converting {path_pt} to {path_onnx}")
    model = torch.jit.load(path_pt)

    model.to(device)
    model.eval()

    batch_size = 10
    input = torch.randn(batch_size, 5, 9, 9)
    outputs = model(input)

    torch.onnx.export(
        model,
        input,
        path_onnx,
        example_outputs=outputs,
        opset_version=12,
        input_names=["input"],
        output_names=["value", "policy"],
        dynamic_axes={"input": {0: "batch_size"}, "value": {0: "batch_size"}, "policy": {0: "batch_size"}},
        training=TrainingMode.EVAL,
    )
