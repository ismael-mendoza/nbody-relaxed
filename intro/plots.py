"""
This file contains classes that represent the different plots that are produced. The purpose is to have more
reproducible plots and separate the plotting procedure from the images produced.
"""
import plotting
import matplotlib.pyplot as plt

class Param(object):
    def __init__(self):
        self.x = x
        self.y = y
        self.cat_name1 = ...  ## also add option to make it log, etc.
        self.cat_name2 = ...
        self.kwargs

class Plot(object):
    """
    Represents a single plot to draw and produce. Each plot will be outputted in a single page of a pdf.
    """

    def __init__(self, title, params, nrows, ncols, figsize, plot_kwargs):
        self.title = title
        self.params = params
        self.fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        self.axes = axs.flatten()

    def run(self, plot_func, **kwargs):
        """
        Produce the plot and save into the axes objects.
        :return: None
        """
        for ax,params in zip(self.axes, self.params):
        plot_func(x, y, )

    def save(self, fname):
        self.fig.savefig(fname)


class ScatterBinning(Plot):
    """
    Represents a single plot to draw and produce. Each plot will be outputted in a single page of a pdf.
    """

    def __init__(self, *args):
        super(ScatterBinning, self).__init__(*args)

    def run(self, **kwargs):
        """
        Produce the plot and save into the axes objects.
        :return: None
        """
        x =
        plotting.scatter_binning(x, y, ax, **kwargs)

    def save(self, fname):
        self.fig.savefig(fname)


