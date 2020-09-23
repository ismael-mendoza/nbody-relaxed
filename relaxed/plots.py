"""This file contains classes that represent the different plots that are produced. The purpose
is to have more reproducible plots and separate the plotting procedure from the images produced.

The parent class 'Plot' only works on the 'high-level', never interacting with the axes objects
directly other than setting them up nad passing them along. The rest is up to plot_funcs.py

It also rounds up all parameter values to be plotted from multiple catalogs and their
corresponding labels.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from relaxed import utils
from . import plot_funcs


class Plot(ABC):
    def __init__(
        self,
        plot_func,
        hparams,
        colors=("r", "b", "g"),
        nrows=1,
        ncols=1,
        figsize=(8, 8),
        title="",
        title_size=20,
        grid_locs=None,
    ):
        """Represents a single plot to draw and produce. Each plot will be outputted
        in a single page of a pdf.

        hparams (list) : A list containing all (unique) halo params necessary for plotting.
        colors (list) : A list of colors to be used when plotting catalogs. len >= different cats.
        """

        self.title = title
        self.title_size = title_size
        self.colors = colors

        self.plot_func = plot_func
        self.hparams = hparams
        self.hparam_dict = {hparam.name: hparam for hparam in self.hparams}
        assert len(self.hparam_dict) == len(self.hparams)

        self.nrows = nrows
        self.ncols = ncols

        self.values = {}

        self._setup_fig_and_axes(grid_locs, figsize)

        # state variables that change when catalog is added.
        self.color_map = {}
        self.n_loaded = 0
        self.hcat_names = []  # preserve order in which catalogs are added.

    def _setup_fig_and_axes(self, grid_locs, figsize):
        # mainly setup grids for plotting multiple axes.

        plt.ioff()

        if not grid_locs:
            # just plot sequentially if locations were not specified.
            self.grid_locs = [
                (i, j) for i in range(self.nrows) for j in range(self.ncols)
            ]
        self.fig, _ = plt.subplots(squeeze=False, figsize=figsize)
        self.axes = [
            plt.subplot2grid((self.nrows, self.ncols), param_loc, fig=self.fig)
            for param_loc in self.grid_locs
        ]

        self.fig.suptitle(self.title, fontsize=self.title_size)

    def save(self, fname=None, pdf=None):
        assert fname or pdf, "one should be specified"
        plt.rc("text", usetex=True)
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if fname:
            self.fig.savefig(utils.figure_path.joinpath(fname))

        else:
            pdf.savefig(self.fig)

    def load(self, hcat):
        """Load the parameter values that will be used for plotting."""
        assert hcat.name not in self.values, "Already loaded."
        assert hcat.name not in self.color_map, "Already loaded"
        self.color_map[hcat.name] = self.colors[self.n_loaded]
        values_i = {}
        for hparam in self.hparams:
            values_i[hparam.name] = hparam.get_values(hcat.cat)
        self.values[hcat.name] = values_i
        self.n_loaded += 1
        self.hcat_names.append(hcat.name)

    @abstractmethod
    def generate(self, plot_params):
        """
        Produce the plot and save into the axes objects. Uses the cached parameters from load
        method.

        Args:
            plot_params (OrderedDict): An ordered dict of single-key dictionaries with order
                                       corresponding to axes each key is param or tuple of
                                       params, values are sets of cat_names.
        """
        pass


class UniPlot(Plot):
    """Creates plot that only depend on one variable at a time, like histograms."""

    def generate(self, plot_params):
        for (ax, param) in zip(self.axes, plot_params):
            for cat_name in plot_params[param]:
                color = self.color_map[cat_name]
                hparam = self.hparam_dict[param]
                param_value = self.values[cat_name][param]
                ax_kwargs = {"xlabel": hparam.text, "use_legend": True}
                self.plot_func(
                    ax,
                    param_value,
                    legend_label=cat_name,
                    color=color,
                    ax_kwargs=ax_kwargs,
                )


class BiPlot(Plot):
    """Class that creates the standard x vs. y plots."""

    def generate(self, plot_params):
        # each param in plot_params is tuple of (param_x, param_y)
        for (ax, param_pair) in zip(self.axes, plot_params):
            for cat_name in plot_params[param_pair]:
                param1, param2 = param_pair
                param1_values = self.values[cat_name][param1]
                param2_values = self.values[cat_name][param2]
                param1_text = self.hparam_dict[param1].text
                param2_text = self.hparam_dict[param2].text
                ax_kwargs = {
                    "xlabel": param1_text,
                    "ylabel": param2_text,
                    "use_legend": True,
                }
                self.plot_func(
                    ax,
                    (param1_values, param2_values),
                    legend_label=cat_name,
                    color=self.color_map[cat_name],
                    ax_kwargs=ax_kwargs,
                )


class MatrixPlot(Plot):
    # Each axes represents a different matrix to be plotted with a different catalog
    # but same hparams.

    def generate(self, plot_params):
        assert len(self.values) == len(
            self.axes
        ), "as many catalogs as axes, since only 1 catalog per axes."

        # collect all params for given cat.
        for (ax, cat_name) in zip(self.axes, self.hcat_names):

            # remember plot_params is an ordered_dict so order is consistent
            # in the following two statements.
            latex_params = [
                self.hparam_dict[param].get_text(only_param=True)
                for param in plot_params
                if cat_name in plot_params[param]
            ]
            values = [
                self.values[cat_name][param]
                for param in plot_params
                if cat_name in plot_params[param]
            ]
            ax_kwargs = {
                "ax_title": cat_name,
                "xticks": range(len(latex_params)),
                "yticks": range(len(latex_params)),
                "xtick_labels": latex_params,
                "ytick_labels": latex_params,
                "use_legend": False,
            }
            self.plot_func(ax, values, ax_kwargs=ax_kwargs)


class Histogram(Plot):
    """Creates histogram and uses caching to set the bin sizes of
    all the catalogs plotted to be the same.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert type(self.plot_func) is plot_funcs.CreateHistogram
        self.create_histogram = self.plot_func
        self.n_bins = self.create_histogram.n_bins

    def run_histogram(self, ax, cat_name, param, color, bins):
        param_value = self.values[cat_name][param]
        hparam = self.hparam_dict[param]
        ax_kwargs = {"use_legend": True, "xlabel": hparam.text}
        self.create_histogram(
            ax,
            param_value,
            ax_kwargs=ax_kwargs,
            bins=bins,
            legend_label=cat_name,
            color=color,
        )

    def generate(self, plot_params):
        for ax, param in zip(self.axes, plot_params):
            param_values = []
            for cat_name in plot_params[param]:
                param_value = self.values[cat_name][param]
                param_values.append(param_value)
            # get the bin edges
            bins = np.histogram(np.hstack(param_values), bins=self.n_bins)[1]

            for cat_name in plot_params[param]:
                color = self.color_map[cat_name]
                self.run_histogram(ax, cat_name, param, color, bins)


class StackedHistogram(Histogram):
    """
    Create a stacked histogram, this is specifically useful to reproduce plots like in Figure 3
    of https://arxiv.org/pdf/1404.4634.pdf, where the top histogram are all the individual plots
    and the bottom row shows the ratio of each with respect to the total.

    * Pass in n_row as if this wasn't stacked (just thinking of normal histogram.
    * Credit:
    https://stackoverflow.com/questions/37737538/merge-matplotlib-subplots-with-shared-x-axis
    """

    # assume the first catalog given is the one we are taking rations with respect to.
    def __init__(self, *args, **kwargs):
        super(StackedHistogram, self).__init__(*args, **kwargs)
        self.main_catalog_idx = 0

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
