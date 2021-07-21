import os

import torch
import torchvision.utils


def make_grid(tensor, repeat: int, padding: int):
    (H, W, h, w) = tensor.shape

    tensor = tensor.repeat_interleave(repeat, 2).repeat_interleave(repeat, 3)
    result = torch.zeros((
        H * (h * repeat + padding) - padding,
        W * (w * repeat + padding) - padding,
    ), dtype=tensor.dtype)

    pass


model = torch.jit.load("../../data/esat/trained_model_10_epochs.pt", map_location="cpu")

os.makedirs("ignored/filters", exist_ok=True)

print("Weight names:")
for key in model.state_dict().keys():
    print(key)
print()

print("BN shapes:")
for key in ["common.1.running_mean", "common.1.running_var", "common.1.num_batches_tracked"]:
    print(f"{key}: {model.state_dict()[key].shape}")
print()

for key in model.state_dict().keys():
    if not key.endswith(".weight"):
        continue
    f = model.state_dict()[key]
    if len(f.shape) != 4 or f.shape[-1] != 3 or f.shape[-2] != 3:
        continue

    print(f"Visualizing {key} with shape {f.shape}")

    output_channels = f.shape[0]
    input_channels = f.shape[1]

    bound = min(-f.quantile(0.01), f.quantile(0.99))

    image = f.transpose(0, 1).reshape(-1, 1, 3, 3).repeat_interleave(16, 2).repeat_interleave(16, 3)
    image = torchvision.utils.make_grid(image, nrow=output_channels, normalize=True, value_range=(-bound, bound),
                                        padding=4)
    image = (image * 255).type(torch.uint8)
    torchvision.io.write_png(image, f"../ignored/filters/{key}.png")

    print((f[0, 0, :, :] + bound) / (2 * bound))
    print((f[0, 1, :, :] + bound) / (2 * bound))
    break
