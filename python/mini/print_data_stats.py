from supervised import print_data_stats
from util import load_data, GoogleData


def main():
    all_data = load_data("../../data/esat3/games_part.csv", shuffle=True)
    test_fraction = 0.02

    split_index = int((1 - test_fraction) * len(all_data))
    train_data = GoogleData.from_generic(all_data.pick_batch(slice(None, split_index)))
    test_data = GoogleData.from_generic(all_data.pick_batch(slice(split_index, None)))

    print_data_stats(test_data, train_data)


if __name__ == '__main__':
    main()
