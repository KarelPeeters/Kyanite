from torch.optim import Adam
from torch.utils.data import DataLoader

from train.module import TrainModule


def bench_limit(module: TrainModule, data: DataLoader, max_steps: int):
    optimizer = Adam(module.parameters())

    for bi, batch in enumerate(data):
        if bi >= max_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss = module.training_step(batch, bi)
        loss.backward()
        optimizer.step()

        print(f"{bi}/{len(data)}: loss {loss.item()}")
