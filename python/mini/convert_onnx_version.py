import torch

from loop import save_onnx

model = torch.jit.load("../../data/derp/good_test_loop/gen_40/model_1_epochs.pt")

save_onnx(model, "../../data/derp/good_test_loop/gen_40/model_1_epochs_v7.onnx")