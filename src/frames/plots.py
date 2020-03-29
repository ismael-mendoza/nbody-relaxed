"""
This file contains classes that represent the different plots that are produced. The purpose is to have more
reproducible plots and separate the plotting procedure from the images produced.
"""
import matplotlib.pyplot as plt
import numpy as np

from ..utils import const


# ToDo: Possibility of merging two plots with two different sets of kwargs, sort of like a matrix merge.
#  it will be annoying to specify kwargs for each of the nrows x ncols but a more global thing
#  ==> make kwargs an attribute of the plot and not just at runtime.

# ToDo: Change to accommodate reading chunkified code.

# ToDo: Mantra to keep in mind: One plot per PDF page.

# ToDo: remove all unnecessary *args, **kwargs!!!

class Plot(object):

    def __init__(self, plot_func, params, param_locs=None, nrows=1, ncols=1, figsize=(8, 8),
                 title='', title_size=20, tick_size=24,
                 plot_kwargs=None):
        """
        Represents a single plot to draw and produce. Each plot will be outputted in a single page of a pdf.

        * To overlay a different plot (say relaxed), just call self.run() again w/ the relaxed catalog and color!
        :param params: Represents a list of :class:`Param`:, could be tuples of params too depending on the plot_func.
        """

        self.title = title
        self.title_size = title_size
        self.tick_size = tick_size

        self.fig, _ = plt.subplots(squeeze=False, figsize=figsize)
        self.nrows = nrows
        self.ncols = ncols

        self.params = params
        self.plot_func = plot_func

        # just plot sequentially if locations were not specified.
        self.param_locs = param_locs if param_locs else [(i, j) for i in range(nrows) for j in range(ncols)]

        self.axes = [plt.subplot2grid((self.nrows, self.ncols), param_loc, fig=self.fig) for param_loc in
                     self.param_locs
                     ]

        self.plot_kwargs = {} if None else plot_kwargs

    def generate(self, cat, *args, **kwargs):
        """
        Produce the plot and save into the axes objects.
        :return: None
        """
        self.preamble()
        self.run(cat, *args, **kwargs, **self.plot_kwargs)
        self.finale()

    def preamble(self):
        self.fig.suptitle(self.title, fontsize=self.title_size)
        plt.ioff()

    def finale(self):

        for ax in self.axes:
            ax.tick_params(axis='both', which='major', labelsize=self.tick_size)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def run(self, *args, **kwargs):
        pass

    def save(self, fname=None, pdf=None):
        plt.rc("text", usetex=True)

        if fname is not None:
            self.fig.savefig(const.figure_path.joinpath(fname))

        elif pdf is not None:
            pdf.savefig(self.fig)

        else:
            raise ValueError("Need to specify either a filename or a pdf")

    # ToDo: Implement some sort of saving necessary cats, args, kwargs before actually plotting
    #       this way we can implement the correct binning procedure:
    #       https://stackoverflow.com/questions/23617129/matplotlib-how-to-make-two-histograms-have-the-same-bin-width/44064402
    #       before plotting combine all values and get the bins using that.
    def load_arguments(self, cat, *args, **kwargs):
        self.cached_arguments.append(cat, args, kwargs)



class BiPlot(Plot):
    """
    Class that creates the standard x vs. y plots.
    """

    def run(self, cat, **kwargs):
        for (ax, param_pair) in zip(self.axes, self.params):
            param1, param2 = param_pair
            self.plot_func(cat, param1, param2, ax, xlabel=param1.text, ylabel=param2.text,
                           **kwargs)


class UniPlot(Plot):
    """
    Creates plot that only depend on one variable at a time, like histograms.
    """

    def run(self, cat, **kwargs):
        for (ax, param) in zip(self.axes, self.params):
            self.plot_func(cat, param, ax, xlabel=param.text, **kwargs)


class StackedHistogram(Plot):
    """
    Create a stacked histogram, this is specifically useful to reproduce plots like in Figure 3 of
    https://arxiv.org/pdf/1404.4634.pdf, where the top histogram are all the individual plots and the bottom row
    shows the ratio of each with respect to the total.
    """


class MatrixPlot(Plot):

    def __init__(self, matrix_func, params, symmetric=False, **kwargs):
        """

        :param matrix_func: A function that returns a matrix of shape len(self.params) x len(self.params).
        :param args:
        """
        self.matrix_func = matrix_func
        self.symmetric = symmetric
        super(MatrixPlot, self).__init__(matrix_func, params, ncols=1, nrows=1, **kwargs)
        self.ax = self.axes[0]

    def run(self, cat, label_size=16, show_cell_text=False, **kwargs):
        matrix = self.matrix_func(self.params, cat)
        mask = np.tri(matrix.shape[0], k=-1) if self.symmetric else None
        a = np.ma.array(matrix, mask=mask)
        im = self.ax.matshow(a, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar(im, ax=self.ax)

        if show_cell_text:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    _ = self.ax.text(j, i, round(matrix[i, j], 2),
                                     ha="center", va="center", color="k", size=14)

        latex_params = [param.get_text(only_param=True) for param in self.params]
        self.ax.set_xticklabels([''] + latex_params, size=label_size)
        self.ax.set_yticklabels([''] + latex_params, size=label_size)
