import numpy as np
from matplotlib import pyplot
from scipy.stats import chisquare


def generate(simulations: int, n: int, s: int):
    """Return an array of size (simulations, options)"""
    a = np.arange(n)
    w = np.ones(n)
    # w[-1] = 0

    s = np.random.choice(a, size=(simulations, s), p=w / w.sum())
    matches = np.arange(n)[None, :, None] == s[:, None, :]
    observations = np.sum(matches, axis=-1)
    return observations


def evaluate(simulations: int, n: int, s: int):
    """Return the mean (p, q) after running many separate X^2 experiments"""
    observations = generate(simulations, n, s)

    _, p = chisquare(observations, axis=-1)
    # p = stats.entropy(observations, np.full_like(observations, 1 / n, dtype=float), axis=-1)

    return np.mean(p)


def main():
    simulations = 1000

    all_n = np.arange(2, 50)
    all_p = []

    for n in all_n:
        p = evaluate(simulations, n, 1000)
        all_p.append(p)
        print(f"{n} -> {p:.4}")

    pyplot.plot(all_n, all_p)
    pyplot.xlabel("N")
    pyplot.ylabel("E[p]")
    pyplot.title("S=1000")
    # pyplot.ylim((0, 1))
    pyplot.show()

    # pyplot.plot(all_n, all_p, label="1-p")
    # pyplot.xlabel("options")
    # pyplot.yscale("log")
    # pyplot.legend()
    # pyplot.show()


if __name__ == '__main__':
    # main()

    data = np.array([3313,
                     3339,
                     3308,
                     3254,
                     3331,
                     3255,
                     3343,
                     3332,
                     3353,
                     3345,
                     3296,
                     3371,
                     3273,
                     3279,
                     3322,
                     3286
                     ])
    print(data.mean())
    x2, p = chisquare(data)

    print(x2, p)
