"""Plotting functions that can be used along with the Plot class in plots.py
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


class PlotFunc(ABC):
    def __init__(
        self,
        xlabel_size=18,
        ylabel_size=18,
        ax_title_size=22,
        legend_size=18,
        tick_size=24,
        tick_label_size=18,
    ):
        self.ax_title_size = ax_title_size
        self.xlabel_size = xlabel_size
        self.ylabel_size = ylabel_size
        self.legend_size = legend_size
        self.tick_size = tick_size
        self.tick_label_size = tick_label_size

    def ax_settings(
        self,
        ax,
        use_legend=False,
        ax_title="",
        xlabel="",
        ylabel="",
        xticks=(),
        yticks=(),
        xtick_labels=(),
        ytick_labels=(),
    ):
        ax.set_title(ax_title, fontsize=self.ax_title_size)
        ax.set_xlabel(xlabel, size=self.xlabel_size)
        ax.set_ylabel(ylabel, size=self.ylabel_size)
        ax.tick_params(axis="both", which="major", labelsize=self.tick_size)

        if use_legend:
            ax.legend(loc="best", prop={"size": self.legend_size})

        if xticks and xtick_labels:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, size=self.tick_label_size)

        if yticks and ytick_labels:
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, size=self.tick_label_size)

    @abstractmethod
    def __call__(self, values, ax, ax_kwargs=None, **plot_kwargs):
        ax_kwargs = {} if ax_kwargs is None else ax_kwargs

        self._plot(values, ax, **plot_kwargs)
        self.ax_settings(ax, **ax_kwargs)

    @abstractmethod
    def _plot(self, ax, values, **kwargs):
        pass


class CreateHistogram(PlotFunc):
    def __init__(
        self, n_bins=30, histtype="step", vline=None, log_y=True, **parent_kwargs
    ):
        super().__init__(**parent_kwargs)
        self.n_bins = n_bins
        self.histtype = histtype
        self.vline = vline
        self.log_y = log_y

    def _plot(self, ax, values, legend_label="", color="r", bins=None, **hist_kwargs):
        """
        Args:
            **hist_kwargs: Additional (general) histogram parameters to plt.hist()
        """
        ax.hist(
            values,
            bins=bins if bins else self.n_bins,
            histtype=self.histtype,
            color=color,
            label=legend_label,
            **hist_kwargs
        )

        # add a vertical line.
        if self.vline == "median":
            ax.axvline(np.median(values), c=color, ls="--")

        if self.log_y:
            ax.set_yscale("log")


# ToDo: Deal with empty bins better, right now it just skips that bin.
class ScatterBinning(PlotFunc):
    def __init__(self, n_xbins=10, show_bands=False, **parent_kwargs):
        super().__init__(**parent_kwargs)
        self.n_xbins = n_xbins
        self.show_bands = show_bands

    def _plot(self, ax, values=(), bin_bds=None, legend_label="", color="r"):
        # values is a tuple values = (x,y)
        x, y = values

        if bin_bds is not None:
            x_bds = np.array(
                [(bin_bds[i], bin_bds[i + 1]) for i in range(len(bin_bds) - 1)]
            )
        else:
            # divide uniformly.
            xs = np.linspace(np.min(x), np.max(x), self.n_xbins)
            x_bds = np.array([(xs[i], xs[i + 1]) for i in range(len(xs) - 1)])

        masks = [((x_bd[0] < x) & (x < x_bd[1])) for x_bd in x_bds]

        xbins = [x[mask] for mask in masks if len(x[mask]) > 0]  # remove empty ones.
        ybins = [y[mask] for mask in masks if len(x[mask]) > 0 and len(y[mask]) > 0]

        xmeds = np.array([np.median(xbin) for xbin in xbins])
        ymeds = np.array([np.median(ybin) for ybin in ybins])

        xdiffs = abs(x_bds.reshape(-1, 2) - xmeds.reshape(-1, 1))

        ax.errorbar(
            xmeds,
            ymeds,
            xerr=xdiffs.T,
            fmt="o-",
            color=color,
            label=legend_label,
            capsize=10,
        )

        y1 = np.array([np.quantile(ybin, 0.25) for ybin in ybins])
        y2 = np.array([np.quantile(ybin, 0.75) for ybin in ybins])

        if self.show_bands:
            ax.fill_between(xmeds, y1, y2, alpha=0.2, linewidth=0.001, color=color)


def spearman_corr(x, y):
    return spearmanr(x, y)[0]


class MatrixValues(PlotFunc):
    def __init__(
        self,
        matrix_func=spearman_corr,
        symmetric=False,
        show_cell_text=False,
        **parent_kwargs
    ):
        """
        Args:
            matrix_func: It is a function that takes maps (values1, values2) --> value.
            symmetric:
            **parent_kwargs:
        """
        super().__init__(**parent_kwargs)
        self.matrix_func = matrix_func
        self.symmetric = symmetric
        self.show_cell_text = show_cell_text

    def _plot(self, ax, values, legend_label=""):
        assert not legend_label, "No legend label for this type of plot."
        # values is a list of tuples (param_name, param_value) in the required order.
        n_params = len(values)
        matrix = np.zeros(n_params, n_params)

        for i, (param1, value1) in enumerate(values):
            for j, (param2, value2) in enumerate(values):
                matrix[i, j] = self.matrix_func(value1, value2)

        # mask out lower off-diagonal elements if requested.
        mask = np.tri(matrix.shape[0], k=-1) if self.symmetric else None
        matrix = np.ma.array(matrix, mask=mask)
        im = ax.matshow(matrix, cmap="bwr", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)

        if self.show_cell_text:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    _ = ax.text(
                        j,
                        i,
                        round(matrix[i, j], 2),
                        ha="center",
                        va="center",
                        color="k",
                        size=14,
                    )
