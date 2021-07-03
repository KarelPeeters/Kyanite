import torch

model = torch.jit.load("../../data/esat3/modest/model_7_epochs.pt", map_location="cpu")

input = torch.zeros(1, 5, 9, 9)
input[0, 0, :, :] = 1

wdl, policy = model(input)

torch.set_printoptions(linewidth=200)
print("WDL:")
print(torch.nn.functional.softmax(wdl, dim=-1))
print("Policy:")
print(torch.nn.functional.softmax(policy.view(-1, 81), dim=-1).view(-1, 9, 9))
