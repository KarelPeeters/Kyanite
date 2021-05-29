import torch.jit

from models import GoogleModel

model = GoogleModel(64, 5, 64)
model = torch.jit.script(model)

model.to("cpu")
model.eval()

batch_size = 17
x_mask = torch.randn(batch_size, 9, 9)
x_tiles = torch.randn(batch_size, 2, 9, 9)
x_macros = torch.randn(batch_size, 2, 3, 3)

torch.onnx.export(
    model,
    (x_mask, x_tiles, x_macros),
    "../data/esat/trained_model_10_epochs.onnx",
    opset_version=10,
    input_names=["mask", "tiles", "macros"],
    output_names=["value", "policy"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
