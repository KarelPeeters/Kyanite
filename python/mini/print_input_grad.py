import torch.jit

torch.set_printoptions(sci_mode=False, linewidth=1000)

x = torch.zeros((1, 3, 7, 7))
x[0, 0, 0, 0] = 1
x[0, 0, 6, 6] = 1
x[0, 1, 0, 6] = 1
x[0, 1, 6, 0] = 1

x = x.clone().detach().requires_grad_(True)

model = torch.jit.load("../../data/ataxx/loop0/gen_11/model_1_epochs.pt", map_location="cpu")

wdl, policy = model(x)
wdl = wdl.softmax(-1)
value = wdl[0, 0] - wdl[0, 2]

value.backward()

print(x.grad)
# print(policy)
