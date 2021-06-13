import torch

from util import o_tensor

o = o_tensor("cpu")
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9])
a = a.view(1, 2, 3, 3)

print(a)
print(a.repeat_interleave(3, 1).repeat_interleave(3, 2))

print(a.view(-1, 9, 1).expand(-1, 9, 9).reshape(-1, 2, 9 * 9)[:, :, o].view(-1, 2, 9, 9))
