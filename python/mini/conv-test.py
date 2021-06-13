import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

device = "cuda"

model = nn.Sequential(
    nn.Linear(8, 1),
)
model.to(device)

optimizer = Adam(model.parameters())
loss_func = MSELoss()

x = torch.rand((1000, 8), device=device)
y = x[:, 0, None]

for epoch in range(1000):
    loss = loss_func(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, loss.item())

for w in model.parameters():
    print(w)
