import matplotlib.pyplot as plt

xpoints = [0.2, 0.4, 0.6]

colors = ['g', 'c', 'm']

for p, c in zip(xpoints, colors):

    plt.axvline(p)

plt.show()