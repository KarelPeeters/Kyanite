from util import load_data

data = load_data("../../data/esat2/all_data.csv", shuffle=False)
print(data.mask_flat.shape)
print(data.mask_flat.sum(axis=1).mean(axis=0))

data = load_data("../../data/loop/games.csv", shuffle=False)
print(data.mask_flat.shape)
print(data.mask_flat.sum(axis=1).mean(axis=0))
