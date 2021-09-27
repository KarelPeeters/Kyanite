from math import prod

import numpy as np
import torch.cuda
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_param_count(module: nn.Module, ):
    param_count = sum(prod(p.shape) for p in module.parameters())
    print(f"Model has {param_count} parameters, which takes {param_count // 1024 / 1024:.3f} Mb")

    for name, child in module.named_children():
        child_param_count = sum(prod(p.shape) for p in child.parameters())
        print(f"  {name}: {child_param_count / param_count:.2f}")


def calc_gradient_norms(module: nn.Module):
    norms = []

    for param in module.parameters():
        if param.grad is not None:
            norms.append((param.grad.detach() ** 2).mean().item())

    return np.array(norms)
