import torch.jit


def print_grid(grid):
    assert len(grid.shape) == 2
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            print("%8.4f" % (20 * grid[row, col].item()), end="")
        print()
    print()


model = torch.jit.load("../../data/esat/trivial_model.pt", map_location="cpu")

r = torch.arange(9)
os = r.view(3, 3).repeat(3, 3)
om = r.view(3, 3).repeat_interleave(3, 0).repeat_interleave(3, 1)
o = (9 * om + os).view(81)

params = list(model.parameters())

print(len(params))
print([p.shape for p in params])

copy = params[0][0, :]


def take(n):
    global copy
    result = copy[:n]
    copy = copy[n:]
    return result


# print_grid(take(81)[o].view(9, 9))
print_grid(take(81)[o].view(9, 9))
print_grid(take(81)[o].view(9, 9))
print_grid(take(9).view(3, 3))
print_grid(take(9).view(3, 3))
assert copy.shape == (0,)
