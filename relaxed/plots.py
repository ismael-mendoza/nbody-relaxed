"""This file contains classes that represent the different plots that are produced. The purpose
is to have more reproducible plots and separate the plotting procedure from the images produced.

The parent class 'Plot' only works on the 'high-level', never interacting with the axes objects
directly other than setting them up nad passing them along. The rest is up to plot_funcs.py

It also rounds up all parameter values to be plotted from multiple catalogs and their
corresponding labels.
"""
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt


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
        figpath="figures",
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

        # where to save figures
        self.figpath = Path(figpath)
        assert self.figpath.exists()

    def _setup_fig_and_axes(self, grid_locs, figsize):
        # mainly setup grids for plotting multiple axes.

        plt.ioff()

        if not grid_locs:
            # just plot sequentially if locations were not specified.
            grid_locs = [(i, j) for i in range(self.nrows) for j in range(self.ncols)]
        self.fig, _ = plt.subplots(squeeze=False, figsize=figsize)
        self.axes = [
            plt.subplot2grid((self.nrows, self.ncols), param_loc, fig=self.fig)
            for param_loc in grid_locs
        ]

        self.fig.suptitle(self.title, fontsize=self.title_size)

    def save(self, fname=None, pdf=None):
        assert fname or pdf, "one should be specified"
        plt.rc("text", usetex=True)
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if fname:
            self.fig.savefig(self.figpath.joinpath(fname))

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
