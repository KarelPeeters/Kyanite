from matplotlib import pyplot

iterations = [100, 300, 1000, 3_000, 10_000, 30_000, 100_000]
sttt_delta_elo = [147.1907, 126.96818, 79.533775, 88.73949, 117.16487, 34.860085, 147.1907]
ataxx_delta_elo = [-407.28995, -190.84851, 52.51155, 240.824, 126.96818, 88.73949, -26.10668]


pyplot.plot(iterations[:len(sttt_delta_elo)], sttt_delta_elo, label="sttt")
pyplot.plot(iterations[:len(ataxx_delta_elo)], ataxx_delta_elo, label="ataxx")

pyplot.xlabel("MCTS nodes")
pyplot.ylabel("ELO improvement of solver")
pyplot.ylim(bottom=0)
pyplot.title("MCTS (solver vs basic) comparison")
pyplot.xscale("log")
pyplot.legend()

pyplot.show()
