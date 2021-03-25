import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

plt.style.use("seaborn-colorblind")


mpl.rcParams.update(
    {
        # figure
        "figure.figsize": (10, 10),
        # axes
        "axes.labelsize": 24,
        "axes.titlesize": 28,
        # ticks
        "xtick.major.size": 5,
        "xtick.minor.size": 2.5,
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "xtick.labelsize": 20,
        "ytick.major.size": 5,
        "ytick.minor.size": 2.5,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "ytick.labelsize": 20,
        # legend
        "legend.fontsize": 24,
    }
)
plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
plt.rc("text", usetex=True)


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
