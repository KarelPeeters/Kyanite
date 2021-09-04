import itertools
import os
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np

from util import uniform_window_filter


def plot_loops(
        paths: [str],
        start_gen: int = 0,
        breaks=None,
        average:bool = False,
        smooth_window: int = 1,
        gen_limit: int = np.inf,
        offsets: Optional[List[int]] = None,
):
    if gen_limit < np.inf:
        print(f"Limiting to {gen_limit} generations")

    breaks = breaks or []

    all_axis = [[] for _ in range(len(paths))]
    all_data = [[] for _ in range(len(paths))]
    gi = None

    for gi in itertools.count(start_gen):
        if gi == gen_limit:
            break
        print(f"Trying gen {gi}")

        gen_axis = []
        gen_data = []

        break_outer = False

        for pi, path in enumerate(paths):
            oi = offsets[pi] if offsets is not None else 0

            try:
                axis = np.load(os.path.join(path, "training", f"gen_{gi + oi}", "plot_axis.npy"))
                data = np.load(os.path.join(path, "training", f"gen_{gi + oi}", "plot_data.npy"))

                if average:
                    gen_axis.append(gi)
                    gen_data.append(data.mean(axis=0))
                else:
                    gen_axis.append(gi + axis)
                    gen_data.append(data)
            except FileNotFoundError:
                break_outer = True
                break

        if break_outer:
            break

        for pi in range(len(paths)):
            all_axis[pi].append(gen_axis[pi])
            all_data[pi].append(gen_data[pi])

    if gi is None:
        print("No data found")
        return

    def add_breaks():
        for b in breaks:
            if start_gen < b < gi:
                plt.axvline(b, color="k")

    for pi in range(len(paths)):
        if average:
            all_axis[pi] = np.array(all_axis[pi])
            all_data[pi] = np.array(all_data[pi])
        else:
            all_axis[pi] = np.concatenate(all_axis[pi])
            all_data[pi] = np.concatenate(all_data[pi])

    all_data = [uniform_window_filter(d, smooth_window, 0) for d in all_data]

    plt.figure()
    for pi in range(len(paths)):
        plt.plot(all_axis[pi], all_data[pi][:, [0, 3]], label=[f"{pi}_train", f"{pi}_test"], alpha=0.5)
    add_breaks()
    plt.legend()
    plt.title("Total")
    plt.show(block=False)

    plt.figure()
    for pi in range(len(paths)):
        plt.plot(all_axis[pi], all_data[pi][:, [1, 4]], label=[f"{pi}_train", f"{pi}_test"], alpha=0.5)
    add_breaks()
    plt.legend()
    plt.title("Value")
    plt.show(block=False)

    plt.figure()
    for pi in range(len(paths)):
        plt.plot(all_axis[pi], all_data[pi][:, [2, 5]], label=[f"{pi}_train", f"{pi}_test"], alpha=0.5)
    add_breaks()
    plt.legend()
    plt.title("Policy")
    plt.show(block=True)

    return gi


if __name__ == '__main__':
    loops = [
        "../data/ataxx/test_loop",
        # "../data/derp/retrain/",
    ]
    plot_loops(loops, start_gen=100, average=True, smooth_window=7, breaks=[330, 379])
