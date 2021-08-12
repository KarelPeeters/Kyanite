import itertools
import os

import matplotlib.pyplot as plt
import numpy as np


def main(path, gen_limit: int = None):
    if gen_limit is not None:
        print(f"Limiting to {gen_limit} generations")

    all_axis = []
    all_data = []
    for gi in itertools.count():
        if gi == gen_limit:
            break

        try:
            all_axis.append(gi + np.load(os.path.join(path, f"gen_{gi}", "plot_axis.npy")))
            all_data.append(np.load(os.path.join(path, f"gen_{gi}", "plot_data.npy")))
        except FileNotFoundError:
            break

    all_axis = np.concatenate(all_axis)
    all_data = np.concatenate(all_data)

    plt.plot(all_axis, all_data[:, [0, 3]])
    plt.legend(["total_train", "total_test"])
    plt.title("Total")
    plt.show()

    plt.plot(all_axis, all_data[:, [1, 4]])
    plt.legend(["value_train", "value_test"])
    plt.title("Value")
    plt.show()

    plt.plot(all_axis, all_data[:, [2, 5]])
    plt.legend(["policy_train", "policy_test"])
    plt.title("Policy")
    plt.show()


if __name__ == '__main__':
    main("../data/derp/test_loop")
