import torch.jit

untrained = torch.jit.load("../data/untrained_model.pt", map_location="cpu")
trained = torch.jit.load("../data/trained_model.pt", map_location="cpu")

torch.set_printoptions(sci_mode=False)

params = list(trained.parameters())

assert len(params) == 2
w = params[0]
b = params[1]

assert b.shape == (1,), b.shape
assert w.shape == (1, 81 + 2 * 81 + 2 * 9), w.shape

w = w[0, :]

print("mask")
print(w.data[:81])

print("next tiles")
print(w.data[81:2 * 81])

print("other tiles")
print(w.data[2 * 81:3 * 81])

print("next macros")
print(w.data[3 * 81:3 * 81 + 9])

print("other macros")
print(w.data[3 * 81 + 9:])

print("bias")
print(b)
