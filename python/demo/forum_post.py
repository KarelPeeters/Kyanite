import numpy as np
import torch

# just write some dummy data
with open("data.txt", "wb") as f:
    f.write(b"test")

# I want to get the file data into this tensor in only a single copy
result = torch.empty(1024)

with open("data.txt", "rb") as f:
    buffer: bytes = f.read()
    array: np.array = np.frombuffer(buffer, dtype=np.float32)
    result[0:4] = array
