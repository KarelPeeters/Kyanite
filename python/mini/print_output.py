import torch.jit


def o_from_xy(x, y):
    return ((x // 3) + (y // 3) * 3) * 9 + ((x % 3) + (y % 3) * 3)


def main():
    model = torch.jit.load("../data/esat/trained_model_5_epochs.pt", map_location="cpu")

    mask = torch.ones(81)
    tiles = torch.zeros(2 * 81)
    macros = torch.zeros(2 * 9)

    tiles[81 + 4 * 9 + 4] = 1

    value, policy_logits = model.forward(mask, tiles, macros)
    policy = torch.softmax(policy_logits, -1)

    policy_xy = torch.zeros(9, 9)
    for y in range(9):
        for x in range(9):
            policy_xy[y, x] = policy[0, o_from_xy(x, y)]

    print(value)
    print(policy_xy)


if __name__ == '__main__':
    main()
