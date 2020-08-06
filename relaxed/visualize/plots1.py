import numpy as np

from . import plot_funcs
import plots
from params import Param
from .. import stats


# ToDo: (Future) some of the redundancy in the functions below can be improved.
def generate_and_save(pdf, hcats, hplots, uplots, colors=None, cached=False):
    if colors is None:
        colors = [None for _ in range(len(hcats))]

    for hcat in hcats:
        assert hcat.get_cat() is not None, "Catalog should be loaded in hcat."

    for hcat, color, hplot in zip(hcats, colors, hplots):

        for plot in hplot:
            if not cached:
                plot.generate(hcat.get_cat(), legend_label=hcat.label, color=color)
            else:
                plot.load_arguments(
                    hcat.get_cat(), legend_label=hcat.label, color=color
                )

    for plot in uplots:
        if cached:
            plot.generate_from_cached()
        if pdf:
            plot.save(pdf=pdf)


def plot_with_mass1(hcats, pdf, colors):
    """
    Catalogs should already be loaded properly.Obtain some of the basic plots where multiple
    catalogs results might be overlaid. This includes:

    * Histograms of Mvir
    * Mean-centered histograms of relevant quantities.
    * Plots demonstrating the correlations between different relaxedness parameters and mvir.
    * (...)

    :param hcats: Already set, list of hcat objects.
    :param colors:
    :param pdf: already prepared pdf object to save figures into.
    :return: None
    """

    # this are the default values that we will be using throughout the plots.
    general_kwargs = dict(xlabel_size=28, ylabel_size=28)
    hist_kwargs = dict(
        bins=30, histtype="step", extra_hist_kwargs=dict(), log_y=True, **general_kwargs
    )

    bin_bds = np.arange(11, 14.5, 0.5)
    binning_kwargs = dict(bin_bds=bin_bds, show_bands=True, **general_kwargs)

    # (1) Need to create all the plots and specify their parameters in kwargss.

    # Plot 1: histogram of Mvir
    params = [Param("mvir", log=True)]
    plot1 = plots.Histogram(
        plot_funcs.histogram, params, nrows=1, ncols=1, plot_kwargs=hist_kwargs
    )

    # Plot 2: Relaxedness parameters and mvir
    relaxed_params = [
        Param("eta", log=True),
        Param("x0", log=True),
        Param("v0", log=True),
        Param("xoff", log=True),
        Param("voff", log=True),
        Param("q", log=True),
        Param("cvir", log=True),
        Param("f_sub", log=False),
        Param("a2", log=True),
    ]
    params = [(Param("mvir", log=True), param) for param in relaxed_params]

    plot2 = plots.BiPlot(
        plot_funcs.scatter_binning,
        params,
        nrows=5,
        ncols=2,
        figsize=(18, 22),
        plot_kwargs=binning_kwargs,
    )

    # (2) Update the unique plots
    uplots = [plot1, plot2]

    # (3) Now specify which hcat to plot in which of the plots via `hplots`.
    # we will use the same plots for both cats, which means we overlay them.
    hplots = [[uplot for uplot in uplots] for _ in hcats]

    # (4) Now, we generate and save all the plots.
    generate_and_save(pdf, hcats, hplots, uplots, colors=colors, cached=True)


def plot_mean_centered_hists(hcats, pdf, colors):
    general_kwargs = dict(xlabel_size=28, ylabel_size=28)

    hist_kwargs2 = dict(
        bins=30,
        histtype="step",
        extra_hist_kwargs=dict(),
        log_y=True,
        vline="median",
        **general_kwargs
    )

    # Plot 2: mean-centered histogram of relevant quantities
    # title = "Mean centered histograms"
    modifiers = [lambda x: (x - np.mean(x)) / np.std(x)]
    params = [
        Param("cvir", log=True, modifiers=modifiers),
        Param("eta", log=True, modifiers=modifiers),
        Param("x0", log=True, modifiers=modifiers),
        Param("v0", log=True, modifiers=modifiers),
        Param("spin", log=True, modifiers=modifiers),
        Param("q", log=True, modifiers=modifiers),
        Param("phi_l", log=True, modifiers=modifiers),
        Param("f_sub", log=False),
    ]
    plot2 = plots.Histogram(
        plot_funcs.histogram,
        params,
        ncols=2,
        nrows=4,
        figsize=(12, 20),
        title=None,
        title_size=24,
        plot_kwargs=hist_kwargs2,
    )

    uplots = [plot2]
    hplots = [[uplot for uplot in uplots] for _ in hcats]
    generate_and_save(pdf, hcats, hplots, uplots, colors=colors, cached=False)


def plot_correlation_matrix_basic(hcats, pdf=None):
    """
    Create a visualization fo the matrix of correlation separate for each of the catalogs in hcats.
    :param hcats:
    :param pdf:
    :return:
    """

    hplots = [[] for _ in hcats]
    uplots = []

    # Plot 4: Matrix correlations
    params = [
        Param("mvir", log=True),
        Param("cvir", log=True),
        Param("eta", log=True),
        Param("x0", log=True),
        Param("v0", log=True),
        Param("spin", log=True),
        Param("q", log=True),
        Param("f_sub", log=False),
        Param("a2", log=True),
        Param("phi_l", log=True),
    ]
    for hcat, hplot in zip(hcats, hplots):
        kwargs = dict(label_size=10, show_cell_text=True)
        plot = plots.MatrixPlot(
            stats.get_corrs,
            params,
            symmetric=False,
            plot_kwargs=kwargs,
            title=hcat.label,
            title_size=24,
        )
        uplots.append(plot)
        hplot.append(plot)

    generate_and_save(pdf, hcats, hplots, uplots)


def plot_decades_basic(hcats, pdf, colors):
    """Produce all the basic plots that require decade separation
    """

    general_kwargs = dict(xlabel_size=28, ylabel_size=28)
    binning_kwargs = dict(n_xbins=8, show_bands=False, **general_kwargs)

    figsize = (24, 24)
    uplots = []
    hplots = [[] for _ in hcats]

    # Plot 5: Correlation between all pairs of different relaxedness parameters as a function
    # of mass half decades.
    # params to include:  't/|u|', 'x0', 'v0', 'xoff', 'Voff', 'q', 'cvir'
    params = [
        (Param("t/|u|", log=True), Param("x0", log=True)),
        (Param("t/|u|", log=True), Param("v0", log=True)),
        (Param("t/|u|", log=True), Param("q", log=True)),
        (Param("t/|u|", log=True), Param("cvir", log=True)),
        (Param("x0", log=True), Param("v0", log=True)),
        (Param("x0", log=True), Param("q", log=True)),
        (Param("x0", log=True), Param("cvir", log=True)),
        (Param("v0", log=True), Param("q", log=True)),
        (Param("v0", log=True), Param("cvir", log=True)),
        (Param("q", log=True), Param("cvir", log=True)),
    ]  # total = 10
    param_locs = []  # triangular pattern, user defined param_locs.
    for i in range(4):
        for j in range(4 - i):
            param_locs.append((j, i))

    plot1 = plots.BiPlot(
        plot_funcs.scatter_binning,
        params,
        nrows=4,
        ncols=4,
        figsize=figsize,
        param_locs=param_locs,
        plot_kwargs=binning_kwargs,
        title_size=40,
    )

    uplots.append(plot1)
    for hplot in hplots:
        hplot.append(plot1)

    generate_and_save(pdf, hcats, hplots, uplots, colors=colors, cached=False)
