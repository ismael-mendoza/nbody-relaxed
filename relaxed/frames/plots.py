"""
This file contains classes that represent the different plots that are produced. The purpose is to have more
reproducible plots and separate the plotting procedure from the images produced.
"""
import matplotlib.pyplot as plt
import numpy as np

from .. import utils

# ToDo: Possibility of merging two plots with two different sets of kwargs, sort of like a matrix merge.
#  it will be annoying to specify kwargs for each of the nrows x ncols but a more global thing
#  ==> make kwargs an attribute of the plot and not just at runtime.

# ToDo: Change to accommodate reading chunkified code.
# ToDo: Mantra to keep in mind: One plot per PDF page.


class Plot(object):
    def __init__(
        self,
        plot_func,
        params,
        param_locs=None,
        nrows=1,
        ncols=1,
        figsize=(8, 8),
        title="",
        title_size=20,
        tick_size=24,
        plot_kwargs=None,
    ):
        """
        Represents a single plot to draw and produce. Each plot will be outputted in a single page of a pdf.

        * To overlay a different plot (say relaxed), just call self.run() again w/ the relaxed catalog and color!
        :param params: Represents a list of :class:`Param`:, could be tuples of params too depending on the plot_func.
        """

        self.title = title
        self.title_size = title_size
        self.tick_size = tick_size

        self.params = params
        self.plot_func = plot_func

        self.nrows = nrows
        self.ncols = ncols

        self.fig, self.axes, self.param_locs = self.get_subplots_config(
            self.nrows, self.ncols, param_locs, figsize
        )

        self.plot_kwargs = {} if None else plot_kwargs
        self.cached_args = []

    def generate(self, cat, *args, **kwargs):
        """
        Produce the plot and save into the axes objects.
        :return: None
        """
        self.preamble()
        self.run(cat, **kwargs, **self.plot_kwargs)
        self.finale()

    def preamble(self):
        self.fig.suptitle(self.title, fontsize=self.title_size)
        plt.ioff()

    def finale(self):

        for ax in self.axes:
            ax.tick_params(axis="both", which="major", labelsize=self.tick_size)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def run(self, cat, **kwargs):
        pass

    def save(self, fname=None, pdf=None):
        plt.rc("text", usetex=True)

        if fname is not None:
            self.fig.savefig(utils.figure_path.joinpath(fname))

        elif pdf is not None:
            pdf.savefig(self.fig)

        else:
            raise ValueError("Need to specify either a filename or a pdf")

    def load_arguments(self, cat, **kwargs):
        self.cached_args.append((cat, kwargs))

    def generate_from_cached(self):
        for (cat, kwargs) in self.cached_args:
            self.generate(cat, **kwargs)

    @staticmethod
    def get_subplots_config(nrows, ncols, param_locs, figsize):
        # just plot sequentially if locations were not specified.
        new_param_locs = (
            param_locs
            if param_locs
            else [(i, j) for i in range(nrows) for j in range(ncols)]
        )

        fig, _ = plt.subplots(squeeze=False, figsize=figsize)
        axes = [
            plt.subplot2grid((nrows, ncols), param_loc, fig=fig)
            for param_loc in new_param_locs
        ]

        return fig, axes, new_param_locs


class BiPlot(Plot):
    """
    Class that creates the standard x vs. y plots.
    """

    def run(self, cat, **kwargs):
        for (ax, param_pair) in zip(self.axes, self.params):
            param1, param2 = param_pair
            self.plot_func(
                cat,
                param1,
                param2,
                ax,
                xlabel=param1.text,
                ylabel=param2.text,
                **kwargs
            )


class UniPlot(Plot):
    """
    Creates plot that only depend on one variable at a time, like histograms.
    """

    def run(self, cat, **kwargs):
        for (ax, param) in zip(self.axes, self.params):
            self.plot_func(cat, param, ax, xlabel=param.text, **kwargs)


class Histogram(UniPlot):
    """
    Creates histograms which is a subclass of UniPlot but uses caching to set the bin sizes of all catalogs
    to be the same.
    """

    def generate_from_cached(self):
        # first we obtain the bin edges.
        assert "bins" in self.plot_kwargs

        num_bins = self.plot_kwargs["bins"]
        bin_edges = []
        for param in self.params:
            param_values = []
            for (cat, _) in self.cached_args:
                param_values.append(param.get_values(cat))

            # get the bin edges
            bins = np.histogram(np.hstack(param_values), bins=num_bins)[1]
            bin_edges.append(bins)

        for (cat, kwargs) in self.cached_args:
            self.generate(cat, bin_edges=bin_edges, **kwargs)

    def run(self, cat, bin_edges=None, **kwargs):
        for i, (ax, param) in enumerate(zip(self.axes, self.params)):
            if bin_edges:
                bin_edge = bin_edges[i]
                assert "bins" in kwargs
                kwargs.update(dict(bins=bin_edge))

            self.plot_func(cat, param, ax, xlabel=param.text, **kwargs)


class StackedHistogram(Histogram):
    """
    Create a stacked histogram, this is specifically useful to reproduce plots like in Figure 3 of
    https://arxiv.org/pdf/1404.4634.pdf, where the top histogram are all the individual plots and the bottom row
    shows the ratio of each with respect to the total.

    * Pass in nrow as if this wasn't stacked (just thinking of normal histogram.
    * Used: https://stackoverflow.com/questions/37737538/merge-matplotlib-subplots-with-shared-x-axis
    """

    def __init__(self, *args, **kwargs):
        super(StackedHistogram, self).__init__(*args, **kwargs)
        self.main_catalog_idx = 0  # assume the first catalog given is the one we are taking rations with respect to.

    # def generate_from_cached(self):
    #     # first get bin edges.
    #     assert 'bins' in self.plot_kwargs
    #
    #     num_bins = self.plot_kwargs['bins']
    #     bin_edges = []
    #     main_cat = self.cached_args[self.main_catalog_idx][0]
    #
    #     for param in self.params:
    #
    #         # first do it normally.
    #         param_values = []
    #         for (cat, _) in self.cached_args:
    #             param_values.append(param.get_values(cat))
    #
    #         bins1 = np.histogram(np.hstack(param_values), bins=num_bins)[1]
    #
    #         # then the ratio ones.
    #         param_values = []
    #         for i, (cat, _) in enumerate(self.cached_args):
    #             if i != self.main_catalog_idx:
    #                 assert len(main_cat) >= len(cat)
    #                 param_values.append(param.get_values(cat) / param.get_values(main_cat))
    #         bins2 = np.histogram(np.hstack(param_values), bins=num_bins)[1]
    #
    #         bin_edges.append((bins1, bins2))
    #
    #     # then use the bin edges to plot.
    #     for (cat, kwargs) in self.cached_args:
    #         self.generate(cat, bin_edges=bin_edges, main_cat=main_cat, **kwargs)
    #
    # @staticmethod
    # def get_subplots_config(nrows, ncols, param_locs, figsize):
    #     fig = plt.figure(figsize=figsize)
    #     new_nrows = nrows*2
    #     grids = gridspec.GridSpec(new_nrows, ncols, height_ratios=[2, 1]*nrows)
    #     axes = [[] for _ in range(new_nrows)]
    #
    #     for i in range(new_nrows):
    #         for j in range(ncols):
    #             gs = grids[i, j]
    #             if i % 2 == 0:
    #                 ax = plt.subplot(gs)
    #             else:
    #                 ax_above = axes[i-1][j]
    #                 ax = plt.subplot(gs, sharex=ax_above)
    #             axes[i].append(ax)
    #     plt.subplots_adjust(hspace=.0)
    #     return fig, axes, param_locs
    #
    # def run(self, cat, bin_edges=None, main_cat=None, **kwargs):
    #     assert main_cat is not None
    #
    #     for i in range(self.nrows*2):
    #         param =
    #         for j in range(self.ncols):
    #             ax = self.axes[i][j]
    #             if bin_edges:
    #                 bin_edge = bin_edges[i][i % 2]
    #                 assert 'bins' in kwargs
    #                 kwargs.update(dict(bins=bin_edge1))
    #             if i % 2 == 0:
    #                 self.plot_func(cat, param, ax, xlabel=param.text, **kwargs)
    #
    #         if i % 2 == 0:
    #             self.plot_func(main_cat, param, )
    #
    #             else:
    #
    #
    #     for i, (ax, param) in enumerate(zip(self.axes, self.params)):
    #         if bin_edges:
    #             bin_edge1, bin_edge2 = bin_edges[i]


class MatrixPlot(Plot):
    def __init__(self, matrix_func, params, symmetric=False, **kwargs):
        """

        :param matrix_func: A function that returns a matrix of shape len(self.params) x len(self.params).
        :param args:
        """
        self.matrix_func = matrix_func
        self.symmetric = symmetric
        super(MatrixPlot, self).__init__(
            matrix_func, params, ncols=1, nrows=1, **kwargs
        )
        self.ax = self.axes[0]

    def run(self, cat, label_size=16, show_cell_text=False, **kwargs):
        matrix = self.matrix_func(self.params, cat)
        mask = np.tri(matrix.shape[0], k=-1) if self.symmetric else None
        a = np.ma.array(matrix, mask=mask)
        im = self.ax.matshow(a, cmap="bwr", vmin=-1, vmax=1)
        plt.colorbar(im, ax=self.ax)

        if show_cell_text:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    _ = self.ax.text(
                        j,
                        i,
                        round(matrix[i, j], 2),
                        ha="center",
                        va="center",
                        color="k",
                        size=14,
                    )

        latex_params = [param.get_text(only_param=True) for param in self.params]
        self.ax.set_xticklabels([""] + latex_params, size=label_size)
        self.ax.set_yticklabels([""] + latex_params, size=label_size)
