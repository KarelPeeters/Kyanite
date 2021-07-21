import onnxruntime
import torch

model = torch.jit.load("../../data/esat3/modest/model_13_epochs.pt", map_location="cpu")

input = torch.zeros(1, 5, 9, 9)
input[0, 0, :, :] = 1

print("Pytorch:")
wdl, policy = model(input)

torch.set_printoptions(linewidth=200)
print(torch.nn.functional.softmax(wdl, dim=-1))
print(torch.nn.functional.softmax(policy.view(-1, 81), dim=-1).view(-1, 9, 9))

print("ONNX:")
sess_options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession("../../data/esat3/modest/model_13_epochs.onnx", sess_options=sess_options)

outputs = session.run(None, {"input": input.numpy()})
onnx_wdl = torch.tensor(outputs[0])
onnx_policy = torch.tensor(outputs[1])
print(torch.nn.functional.softmax(onnx_wdl, dim=-1))
print(torch.nn.functional.softmax(onnx_policy.view(-1, 81), dim=-1).view(-1, 9, 9))
