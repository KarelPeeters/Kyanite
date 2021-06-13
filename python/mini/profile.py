import torch.jit
from torch.profiler import tensorboard_trace_handler

from models import GoogleModel
from util import load_data, o_tensor

batch_size = 1000
device = "cuda"

model = GoogleModel(64, 4, 8, 64, True)
model = torch.jit.script(model)
model.eval()
model.to(device)
data = load_data("../../data/esat/test_data.csv")

batch = data.pick_batch(torch.arange(batch_size)).to(device)
o = o_tensor(device)

print("Kineto available:", torch.autograd.kineto_available())

with torch.profiler.profile(
        # schedule=torch.profiler.schedule(
        #     wait=2,
        #     warmup=2,
        #     active=6,
        #     repeat=1,
        # ),
        on_trace_ready=tensorboard_trace_handler("foobar"),
        # with_stack=True,
        # use_cuda=True,
) as profiler:
    for _ in range(10):
        _ = model(o, batch.mask, batch.x_tiles, batch.x_macros)
        profiler.step()
        print("step done")

print("done")
