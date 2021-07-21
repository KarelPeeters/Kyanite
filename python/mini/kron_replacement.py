import torch

range = torch.arange(9)
os = range.view(3, 3).repeat(3, 3)
om = range.view(3, 3).repeat_interleave(3, 0).repeat_interleave(3, 1)
o = (9 * om + os).view(9, 9)

print(o)