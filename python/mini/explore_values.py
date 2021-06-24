import torch

from util import load_data, GoogleData

data = load_data("../../data/esat2/all_data.csv")

indices = torch.randint(len(data), (20,))
data = data.pick_batch(indices)

google = GoogleData.from_generic(data)
mean = (google.value + google.final_value) / 2

options = {
    "value": google.value,
    "final": google.final_value,
    "mean": mean
}

for name, opt in options.items():
    print(f"{name}:")
    print("  values:    ", opt[:8].flatten())
    print("  std, mean: ", torch.std_mean(opt))
    print("  var, mean: ", torch.var_mean(opt))
