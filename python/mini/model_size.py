from math import prod

from models import GoogleModel


def count_params(model):
    return sum(prod(param.shape) for param in model.parameters())


param_count = count_params(GoogleModel(channels=64, blocks=3, wdl_size=64, res=False))


print(f"param count: {param_count}")
print(f"model size: {param_count // 1024 / 1024:.3}MB")
