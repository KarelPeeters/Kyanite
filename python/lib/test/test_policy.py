import torch

from lib.data.file import DataFile
from lib.data.position import PositionBatch
from lib.games import Game
from lib.train import new_evaluate_policy, evaluate_policy


def check_equal(actual, expected):
    print(f"{actual} == {expected}")
    assert actual == expected


def empty(device, mask):
    result = new_evaluate_policy(
        torch.zeros((5, 6), dtype=torch.float32, device=device),
        torch.zeros((5, 0), dtype=torch.int64, device=device),
        torch.zeros((5, 0), dtype=torch.float32, device=device),
        mask_invalid_moves=mask,
    )
    # print(result)

    # check_equal(
    #     loss,
    #     nnf.cross_entropy(
    #         torch.tensor([0.0, 0.0]).unsqueeze(0),
    #         torch.tensor([0.5, 0.5]).unsqueeze(0),
    #         reduce=False
    #     )
    # )


def basic(device, mask):
    result = new_evaluate_policy(
        torch.tensor([[0.0, 0.0, -20.0, -100.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]], device=device),
        torch.tensor([[0, 1], [0, 0], [0, 0]], device=device),
        torch.tensor([[0.5, 0.5], [1.0, -1.0], [-1.0, -1.0]], device=device),
        mask_invalid_moves=mask,
    )
    # print(result)

    # check_equal(
    #     loss,
    #     nnf.cross_entropy(
    #         torch.tensor([0.0, 0.0]).unsqueeze(0),
    #         torch.tensor([0.5, 0.5]).unsqueeze(0),
    #         reduce=False
    #     )
    # )


def test_combinations():
    for mask in [False, True]:
        for device in ["cpu", "cuda"]:
            print(f"============ Mask: {mask}, device: {device}")
            basic(device, mask)
            print()
            empty(device, mask)
            print()
            print()


def compare_ataxx():
    game = Game.find("ataxx-6")
    file = DataFile.open(game, "D:/Documents/A0/loop/ataxx-6/more-exploration/selfplay/games_136.json")

    # for i in [0, 10, 541]:
    for i in range(len(file) // 2):
        p0 = file[2 * i]
        p1 = file[2 * i + 1]

        print(f"{i}: available moves: {p0.available_mv_count} {p1.available_mv_count}")

        if p0.available_mv_count == 0 or p1.available_mv_count == 0:
            print(f"No available moves: ")
        if p0.available_mv_count != p1.available_mv_count:
            print("Padded policy indices")

        batch = PositionBatch(game, [p0, p1], pin_memory=False)

        logits = torch.randn(len(batch), *game.policy_shape)

        indices = batch.policy_indices.to("cpu")
        values = batch.policy_values.to("cpu")

        old_loss, _, _ = evaluate_policy(logits, indices, values)

        masked = new_evaluate_policy(logits, indices, values, True)
        unmasked = new_evaluate_policy(logits, indices, values, False)

        print(f"{old_loss.item()} =>")
        print(f"  {masked}")
        print(f"  {unmasked}")

        if i > 2000:
            break


def main():
    compare_ataxx()


if __name__ == '__main__':
    main()
