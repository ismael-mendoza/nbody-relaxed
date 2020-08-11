"""
Plotting functions that are useful for visualizing things like correlations.
"""

import numpy as np

from params import params


def general_ax_settings(
    ax,
    ax_title="",
    xlabel=None,
    ylabel=None,
    xlabel_size=18,
    ylabel_size=18,
    legend_label=None,
    legend_size=18,
    title_size=22,
):
    ax.set_title(ax_title, fontsize=title_size)

    if xlabel is not None:
        ax.set_xlabel(xlabel, size=xlabel_size)

    if ylabel is not None:
        ax.set_ylabel(ylabel, size=ylabel_size)

    if legend_label:
        ax.legend(loc="best", prop={"size": legend_size})


def histogram(
    cat,
    param,
    ax,
    bins=30,
    histtype="step",
    color="r",
    legend_label=None,
    vline=None,
    log_y=True,
    extra_hist_kwargs=None,
    **general_kwargs
):
    if extra_hist_kwargs is None:
        extra_hist_kwargs = {}

    values = param.get_values(cat)
    ax.hist(
        values,
        bins=bins,
        histtype=histtype,
        color=color,
        label=legend_label,
        **extra_hist_kwargs
    )

    # add a vertical line.
    if vline == "median":
        ax.axvline(np.median(values), c=color, ls="--")

    # log the scale or not.
    if log_y:
        ax.set_yscale("log")

    general_ax_settings(ax, legend_label=legend_label, **general_kwargs)


def binning3d_mass(
    cat,
    param1,
    param2,
    ax,
    ax_title=None,
    mass_decades=np.arange(11, 15, 1),
    **scatter_binning_kwargs
):
    mass_bins = [(x, y) for x, y in zip(mass_decades, mass_decades[1:])]
    colors = ["b", "r", "g"]
    for mass_bin, color in zip(mass_bins, colors):
        log_mvir = params.HaloParam("mvir", log=True).get_values(cat)
        mmask = (log_mvir > mass_bin[0]) & (log_mvir < mass_bin[1])
        mcat = cat[mmask]
        label = "$" + str(mass_bin[0]) + "< M_{\\rm vir} <" + str(mass_bin[1]) + "$"

        # avoid conflict with legend_label inside kwargs.
        scatter_binning_kwargs.update(dict(legend_label=label, color=color))

        scatter_binning(
            mcat, param1, param2, ax_title=ax_title, ax=ax, **scatter_binning_kwargs
        )


def scatter_binning(
    cat,
    param1,
    param2,
    ax,
    n_xbins=10,
    bin_bds=None,
    color="r",
    show_bands=False,
    xlabel=None,
    ylabel=None,
    legend_label=None,
    **general_kwargs
):
    # ToDo: Deal with empty bins better, right now it just skips that bin.
    x = param1.get_values(cat)
    y = param2.get_values(cat)

    if bin_bds is not None:
        x_bds = np.array(
            [(bin_bds[i], bin_bds[i + 1]) for i in range(len(bin_bds) - 1)]
        )
    else:
        xs = np.linspace(np.min(x), np.max(x), n_xbins)
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

    if show_bands:
        ax.fill_between(xmeds, y1, y2, alpha=0.2, linewidth=0.001, color=color)

    general_ax_settings(
        ax, xlabel=xlabel, ylabel=ylabel, legend_label=legend_label, **general_kwargs
    )
