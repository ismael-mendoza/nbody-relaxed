"""
Plotting functions that are useful for visualizing things like correlations.
"""

import numpy as np
from src.frames import params


def general_ax_settings(ax, ax_title='', xlabel=None, ylabel=None, xlabel_size=18, ylabel_size=18, legend_label=None,
                        legend_size=18, title_size=22):

    ax.set_title(ax_title, fontsize=title_size)

    if xlabel is not None:
        ax.set_xlabel(xlabel, size=xlabel_size)

    if ylabel is not None:
        ax.set_ylabel(ylabel, size=ylabel_size)

    if legend_label:
        ax.legend(loc='best', prop={'size': legend_size})


def histogram(cat, param, ax, bins=30, histtype='step', color='r', legend_label=None, extra_hist_kwargs=None,
              **general_kwargs):

    if extra_hist_kwargs is None:
        extra_hist_kwargs = {}

    values = param.get_values(cat)
    ax.hist(values, bins=bins, histtype=histtype, color=color, label=legend_label, **extra_hist_kwargs)

    general_ax_settings(ax, legend_label=legend_label, **general_kwargs)


def binning3d_mass(cat, param1, param2, ax, ax_title=None, mass_decades=np.arange(11, 15, 1),
                   **scatter_binning_kwargs):
    mass_bins = [(x, y) for x, y in zip(mass_decades, mass_decades[1:])]
    colors = ['b', 'r', 'g']
    for mass_bin, color in zip(mass_bins, colors): 
        log_mvir = params.Param('mvir', log=True).get_values(cat)
        mmask = (log_mvir > mass_bin[0]) & (log_mvir < mass_bin[1])
        mcat = cat[mmask]
        label = "$" + str(mass_bin[0]) + "< M_{\\rm vir} <" + str(mass_bin[1]) + "$"

        # avoid conflict with legend_label inside kwargs.
        scatter_binning_kwargs.update(dict(legend_label=label, color=color))

        scatter_binning(mcat,
                        param1, param2, ax_title=ax_title,
                        ax=ax, **scatter_binning_kwargs)


def scatter_binning(cat, param1, param2, ax, nxbins=10, color='r', no_bars=False, show_lines=False, show_bands=False,
                    xlabel=None, ylabel=None, legend_label=None, **general_kwargs):
    # ToDo: Deal with empty bins better, right now it just skips that bin.
    x = param1.get_values(cat)
    y = param2.get_values(cat)

    xs = np.linspace(np.min(x), np.max(x), nxbins)
    xbbins = [(xs[i], xs[i+1]) for i in range(len(xs)-1)]

    masks = [((xbbin[0] < x) & (x < xbbin[1])) for xbbin in xbbins]

    binned_x = [x[mask] for mask in masks if len(x[mask]) > 0]  # remove empty ones.
    binned_y = [y[mask] for mask in masks if len(x[mask]) > 0 and len(y[mask]) > 0]

    xmeds = [np.median(xbin) for xbin in binned_x]
    ymeds = [np.median(ybin) for ybin in binned_y]

    xqs = np.array([[xmed - np.quantile(xbin, 0.25), np.quantile(xbin, 0.75) - xmed] for (xmed, xbin)
                    in zip(xmeds, binned_x)]).T
    yqs = np.array([[ymed - np.quantile(ybin, 0.25), np.quantile(ybin, 0.75) - ymed] for (ymed, ybin)
                    in zip(ymeds, binned_y)]).T

    if not no_bars:
        ax.errorbar(xmeds, ymeds, xerr=xqs, yerr=yqs, fmt='o--', capsize=10, color=color, label=legend_label)
    else:
        ax.errorbar(xmeds, ymeds, xerr=xqs, fmt='o-', color=color, label=legend_label, capsize=10)

    y1 = np.array([np.quantile(ybin, 0.25) for ybin in binned_y])
    y2 = np.array([np.quantile(ybin, 0.75) for ybin in binned_y])

    if show_lines:
        ax.plot(xmeds, y1, '--', color=color)
        ax.plot(xmeds, y2, '--', color=color)

    if show_bands:
        ax.fill_between(xmeds, y1, y2, alpha=0.2, linewidth=0.001, color=color)

    general_ax_settings(ax, xlabel=xlabel, ylabel=ylabel, legend_label=legend_label, **general_kwargs)


