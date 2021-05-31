import math

import numpy as np
from matplotlib import pyplot


def plot(games, temp_move_count: int):
    thresholds = np.linspace(-0.25, -.95, 100)
    wrong_rate = np.zeros(len(thresholds))
    moves_fraction_saved = np.zeros(len(thresholds))

    for ti, t in enumerate(thresholds):
        wrong_count = 0
        save_count = 0
        total_move_count = 0

        for gi, (scores, result) in enumerate(games):
            resign_expected_winner = 0
            total_move_count += len(scores)

            factor = 1
            for si, score in enumerate(scores):
                if factor * score <= t:
                    resign_expected_winner = factor
                    save_count += len(scores) - si
                    break

                factor *= -1

            # print(f"game {gi} expected winner {resign_expected_winner} actual {result}")
            wrong_count += resign_expected_winner != result

        wrong_rate[ti] = wrong_count / len(games)
        moves_fraction_saved[ti] = save_count / total_move_count

    for g in games:
        pyplot.plot(g[0], "k", alpha=0.05)
    pyplot.title("Game progress")
    pyplot.xlabel("move")
    pyplot.ylabel("value")
    if not math.isnan(temp_move_count):
        pyplot.axvline(temp_move_count, -1, 1, c="r", label="start playing optimal moves")
    pyplot.legend()
    pyplot.show()

    pyplot.plot(thresholds, wrong_rate, label="wrong resign rate")
    pyplot.plot(thresholds, moves_fraction_saved, label="fraction of computation time saved")
    pyplot.legend()
    pyplot.title("Resigning if the estimated value is too low")
    pyplot.xlabel("resignation threshold")
    pyplot.ylim(0, 0.5)
    pyplot.xlim(-1, np.max(thresholds))
    pyplot.show()


def main():
    from mini.resignation_threshold_data import RESIGNATION_DATA_ALWAYS_POLICY
    from mini.resignation_threshold_data import RESIGNATION_DATA_OPTIMAL_20

    min_length = min(len(RESIGNATION_DATA_ALWAYS_POLICY), len(RESIGNATION_DATA_OPTIMAL_20))

    plot(RESIGNATION_DATA_ALWAYS_POLICY[:min_length], np.nan)
    plot(RESIGNATION_DATA_OPTIMAL_20[:min_length], 20)


if __name__ == '__main__':
    main()
