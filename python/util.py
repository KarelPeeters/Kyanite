import numpy as np
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")


def linspace_int(stop: int, num: int) -> np.array:
    assert 0 < stop and 0 < num

    if num >= stop:
        return np.arange(stop)
    else:
        return np.linspace(0, stop - 1, num).astype(int)


def uniform_window_filter(data: np.array, window_size: int, axis: int) -> np.array:
    if len(data) <= window_size * 3 or window_size < 2:
        return data

    from scipy.signal import filtfilt
    a = 1
    b = [1 / window_size] * window_size
    return filtfilt(b, a, data, axis=axis)


def prod(i):
    """Equivalent to the existing math.prod function but works in older python versions (as found on eg. AWS)"""
    total = 1
    for x in i:
        total *= x
    return total


def print_param_count(module: nn.Module, f=None):
    param_count = sum(prod(p.shape) for p in module.parameters())
    print(f"Model has {param_count} parameters, which takes {param_count // 1024 / 1024:.3f} Mb", file=f)
    for name, child in module.named_children():
        child_param_count = sum(prod(p.shape) for p in child.parameters())
        print(f"  {name}: {child_param_count / param_count:.2f}", file=f)
