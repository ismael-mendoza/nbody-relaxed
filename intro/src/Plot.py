"""
This file contains classes that represent the different plots that are produced. The purpose is to have more
reproducible plots and separate the plotting procedure from the images produced.
"""
import matplotlib.pyplot as plt
from pathlib import Path

plot_directory=

#ToDo: Need to make it more flexible to accomodate graphs like the covariance one.

class Plot(object):

    def __init__(self, title, params, nrows, ncols, figsize):
        """
        Represents a single plot to draw and produce. Each plot will be outputted in a single page of a pdf.
        :param params: Represents a list of :class:`Param`: in the order of axs.flatten() that will be plotted.
        :param xs: Range of x to plot the params.
        """

        self.title = title
        self.params = params
        self.fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        self.axes = axs.flatten()

    def run(self, plot_func, **kwargs):
        """
        Produce the plot and save into the axes objects.
        :return: None
        """
        plt.ioff()
        plt.rc("text", usetex=True)

        for (ax, param_pair) in zip(self.axes, self.params):
            param1, param2 = param_pair
            plot_func(param1.get_values(), param2.get_values(), ax, xlabel=param1.text, ylabel=param2.text, **kwargs)
        self.fig.tight_layout()

    def save(self, fname):
        self.fig.savefig(fname)


