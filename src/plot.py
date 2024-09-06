import matplotlib.pyplot as plt
import numpy as np


def signal_plot_setup(ax):
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xticks([np.pi * i for i in range(3)], ['$0$', '$\pi$', '$2\pi$'])
    