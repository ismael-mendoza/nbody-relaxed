"""
This file contains classes that represent the different plots that are produced. The purpose is to
have more reproducible plots and separate the plotting procedure from the images produced.
"""
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from relaxed import utils


def general_ax_settings(
    ax,
    ax_title="",
    xlabel="",
    ylabel="",
    legend_label="",
    xlabel_size=18,
    ylabel_size=18,
    legend_size=18,
    ax_title_size=22,
):
    ax.set_title(ax_title, fontsize=ax_title_size)
    ax.set_xlabel(xlabel, size=xlabel_size)
    ax.set_ylabel(ylabel, size=ylabel_size)

    if legend_label:
        ax.legend(loc="best", prop={"size": legend_size})


class Plot(ABC):
    def __init__(
        self,
        plot_func,
        hparams,
        nrows=1,
        ncols=1,
        figsize=(8, 8),
        title="",
        title_size=20,
        tick_size=24,
        grid_locs=None,
        plot_kwargs=None,
    ):
        """Represents a single plot to draw and produce. Each plot will be outputted
        in a single page of a pdf.

        hparams (list) : A list containing all unique halo params necessary for plotting.
        """

        self.title = title
        self.title_size = title_size
        self.tick_size = tick_size

        self.plot_func = plot_func
        self.hparams = hparams
        self.hparam_dict = {hparam.name: hparam for hparam in self.hparams}
        assert len(self.hparam_dict) == len(self.hparams)

        self.nrows = nrows
        self.ncols = ncols

        self.plot_kwargs = {} if None else plot_kwargs
        self.values = {}

        self._setup_fig_and_axes(grid_locs, figsize)

    @staticmethod
    def _get_subplots_config(nrows, ncols, grid_locs, figsize):
        # just plot sequentially if locations were not specified.
        if not grid_locs:
            grid_locs = [(i, j) for i in range(nrows) for j in range(ncols)]

        fig, _ = plt.subplots(squeeze=False, figsize=figsize)
        axes = [
            plt.subplot2grid((nrows, ncols), param_loc, fig=fig)
            for param_loc in grid_locs
        ]

        return fig, axes, grid_locs

    def _setup_fig_and_axes(self, grid_locs, figsize):
        # setup figure and axes
        plt.ioff()
        self.fig, self.axes, self.grid_locs = self._get_subplots_config(
            self.nrows, self.ncols, grid_locs, figsize
        )
        self.fig.suptitle(self.title, fontsize=self.title_size)
        for ax in self.axes:
            ax.tick_params(axis="both", which="major", labelsize=self.tick_size)
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def save(self, fname=None, pdf=None):
        assert fname or pdf, "one should be specified"
        plt.rc("text", usetex=True)

        if fname:
            self.fig.savefig(utils.figure_path.joinpath(fname))

        else:
            pdf.savefig(self.fig)

    def load(self, hcat):
        """Load the parameter values that will be used for plotting."""
        # idx = idx of catalog added.
        assert hcat.name not in self.values, "Cat already loaded."
        values_i = {}
        for hparam in self.hparams:
            values_i[hparam.name] = hparam.get_values(hcat.cat)
        self.values[hcat.name] = values_i

    def generate(self, plot_params, **kwargs):
        """
        Produce the plot and save into the axes objects.
        :return: None
        """
        self._run(plot_params, **kwargs, **self.plot_kwargs)

    @abstractmethod
    def _run(self, plot_params, **kwargs):
        """Actually create the plots.
        Args:
            plot_params: Always a collection of strings.
        """
        pass


class BiPlot(Plot):
    """Class that creates the standard x vs. y plots."""

    def _run(self, plot_params, **kwargs):
        # plot_params = [(param11, param12), (param21,param22)...] (strings)
        for cat_name in self.values:
            for (ax, param_pair) in zip(self.axes, plot_params):
                param1, param2 = param_pair
                param1_values = self.values[cat_name][param1]
                param2_values = self.values[cat_name][param2]
                param1_text = self.hparam_dict[param1].text
                param2_text = self.hparam_dict[param2].text
                self.plot_func(
                    param1_values,
                    param2_values,
                    ax,
                    xlabel=param1_text,
                    ylabel=param2_text,
                    **kwargs
                )


class UniPlot(Plot):
    """Creates plot that only depend on one variable at a time, like histograms."""

    def _run(self, plot_params, **kwargs):
        for cat_name in self.values:
            for (ax, param) in zip(self.axes, plot_params):
                param_value = self.values[cat_name][param]
                self.plot_func(param_value, ax, xlabel=param.text, **kwargs)


class MatrixPlot(Plot):
    def __init__(self, matrix_func, hparams, symmetric=False, **kwargs):
        super().__init__(matrix_func, hparams, ncols=1, nrows=1, **kwargs)
        self.matrix_func = matrix_func
        self.symmetric = symmetric
        self.ax = self.axes[0]

    def _run(self, label_size=16, show_cell_text=False, **kwargs):
        assert len(self.values) == 1
        name, values = self.values.popitem()
        matrix = self.matrix_func(values)

        # mask out lower off-diagonal elements if requested.
        mask = np.tri(matrix.shape[0], k=-1) if self.symmetric else None
        matrix = np.ma.array(matrix, mask=mask)
        im = self.ax.matshow(matrix, cmap="bwr", vmin=-1, vmax=1)
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
        self.ax.set_xticks(range(len(latex_params)))
        self.ax.set_yticks(range(len(latex_params)))

        self.ax.set_xticklabels(latex_params, size=label_size)
        self.ax.set_yticklabels(latex_params, size=label_size)


class Histogram(UniPlot):
    """Creates histograms which is a subclass of UniPlot, but uses caching to set the bin sizes of
    all the catalogs plotted to be the same.
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
    Create a stacked histogram, this is specifically useful to reproduce plots like in Figure 3
    of https://arxiv.org/pdf/1404.4634.pdf, where the top histogram are all the individual plots
    and the bottom row shows the ratio of each with respect to the total.

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
