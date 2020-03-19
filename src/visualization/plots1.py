from src.frames import plots
from src.frames.params import Param
from src.visualization import plot_funcs
from src import stats

import numpy as np

# ToDo: (Future) some of the redundancy in the functions below can be improved.


def generate_and_save(pdf, hcats, argss, uplots, colors=None):
    if colors is None:
        colors = [None for _ in range(len(hcats))]

    for hcat, color, args in zip(hcats, colors, argss):
        for plot, kwargs in args:
            plot.generate(hcat.get_cat(),
                          legend_label=hcat.catalog_label, color=color,
                          **kwargs)

    for plot in uplots:
        plot.save(pdf=pdf)


def plot_multiple_basic(hcats, pdf, colors):
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

    for hcat in hcats:
        assert hcat.get_cat() is not None, "Catalog should be loaded in hcat."

    # this are the default values that we will be using throughout the plots.
    general_kwargs = dict(xlabel_size=28, ylabel_size=28)
    hist_kwargs = dict(bins=30, histtype='step', extra_hist_kwargs=dict(), **general_kwargs)
    binning_kwargs = dict(nxbins=10, no_bars=True, show_bands=True, **general_kwargs)

    # (1) Need to create all the plots and specify their parameters in kwargss.
    uplots = []

    # Plot 1: histogram of Mvir
    params = [Param('mvir', log=True)]
    plot1 = plots.UniPlot(plot_funcs.histogram, params, nrows=1, ncols=1)

    # Plot 2: mean-centered histogram of relevant quantities
    # ToDo: Fix the title to make it a possibility.
    # title = "Mean centered histograms"
    modifiers = [lambda x: (x - np.mean(x)) / np.std(x)]
    param_names = ['mvir', 'cvir', 'T/|U|', 'xoff', 'voff', 'Spin', 'q', 'phi_l']
    params = [Param(param_name, log=True, modifiers=modifiers) for param_name in param_names]
    plot2 = plots.UniPlot(plot_funcs.histogram, params, ncols=2, nrows=4, figsize=(12, 20),
                          title=None, title_size=24)

    # Plot 3: Relaxedness parameters and mvir
    relaxedness_param_names = ['T/|U|', 'xoff', 'voff', 'Xoff', 'Voff', 'q', 'cvir']
    params = [(Param('mvir', log=True), Param(relaxed_param, log=True)) for relaxed_param in relaxedness_param_names]
    plot3 = plots.BiPlot(plot_funcs.scatter_binning, params, nrows=3, ncols=2, figsize=(18, 18))

    # (2) Update the unique plots
    uplots.extend([plot1, plot2, plot3])

    # (3) Now specify which hcat to plot in which of the plots via argss.
    argss = [[] for _ in hcats]  # list corresponding to a cat of list of tuple of (plot, kwargss)

    for args in argss:  # one args for each hcats.
        # we will use the same plots for both cats, which means we overlay them.
        args.append(
            (plot1, hist_kwargs)
        )

        args.append(
            (plot2, hist_kwargs)
        )

        args.append(
            (plot3, binning_kwargs)
        )

    # (4) Now, we generate and save all the plots.
    generate_and_save(pdf, hcats, argss, uplots, colors=colors)


def plot_correlation_matrix_basic(hcats, pdf):
    """
    Create a visualization fo the matrix of correlation separate for each of the catalogs in hcats.
    :param hcats:
    :param pdf:
    :return:
    """
    for hcat in hcats:
        assert hcat.get_cat() is not None, "Catalog should be loaded in hcat."

    argss = [[] for _ in hcats]
    uplots = []

    # Plot 4: Matrix correlations
    param_names = ['mvir', 'cvir', 'T/|U|', 'xoff', 'voff', 'Spin', 'q', 'phi_l']
    params = [Param(param_name, log=True) for param_name in param_names]
    for args in argss:
        plot = plots.MatrixPlot(stats.get_corrs, params, symmetric=False)
        kwargs = dict(label_size=20)
        uplots.append(plot)
        args.append(
           (plot, kwargs)
        )

    generate_and_save(pdf, hcats, argss, uplots)


def plot_decades_basic(hcats, pdf):
    """
    Produce all the basic plots that require decade separation
    :return:
    """
    for hcat in hcats:
        assert hcat.get_cat() is not None, "Catalog should be loaded in hcat."

    general_kwargs = dict(xlabel_size=28, ylabel_size=28)
    binning_3d_kwargs = dict(nxbins=8, no_bars=False, show_bands=False, **general_kwargs)
    binning_kwargs = dict(nxbins=8, no_bars=True, show_bands=True, **general_kwargs)

    figsize = (24, 24)
    uplots = []
    argss = [[] for _ in hcats]

    # Plot 5: Correlation between all pairs of different relaxedness parameters as a function of mass half decades.
    # params to include:  'T/|U|', 'xoff', 'voff', 'Xoff', 'Voff', 'q', 'cvir'
    params = [
        (Param('T/|U|', log=True), Param('xoff', log=True)),
        (Param('T/|U|', log=True), Param('voff', log=True)),
        (Param('T/|U|', log=True), Param('Xoff', log=True)),
        (Param('T/|U|', log=True), Param('Voff', log=True)),
        (Param('T/|U|', log=True), Param('q', log=True)),
        (Param('T/|U|', log=True), Param('cvir', log=True)),

        (Param('xoff', log=True), Param('voff', log=True)),
        (Param('xoff', log=True), Param('Voff', log=True)),
        (Param('xoff', log=True), Param('q', log=True)),
        (Param('xoff', log=True), Param('cvir', log=True)),

        (Param('Xoff', log=True), Param('voff', log=True)),
        (Param('Xoff', log=True), Param('Voff', log=True)),
        (Param('Xoff', log=True), Param('q', log=True)),
        (Param('Xoff', log=True), Param('cvir', log=True)),

        (Param('voff', log=True), Param('Voff', log=True)),
        (Param('voff', log=True), Param('q', log=True)),
        (Param('voff', log=True), Param('cvir', log=True)),

        (Param('Voff', log=True), Param('q', log=True)),
        (Param('Voff', log=True), Param('cvir', log=True)),

        (Param('q', log=True), Param('cvir', log=True)),
        ]  # total = 20

    plot1 = plots.BiPlot(plot_funcs.binning3d_mass, params, nrows=5, ncols=4, figsize=figsize)
    uplots.append(plot1)

    for args in argss:
        args.append(
            (plot1, binning_3d_kwargs)
        )

    # Plot 6: Same plot as above but without any decades.
    plot2 = plots.BiPlot(plot_funcs.scatter_binning, params, nrows=5, ncols=4, figsize=figsize)
    uplots.append(plot2)

    for args in argss:
        args.append(
            (plot2, binning_kwargs)
        )

    generate_and_save(pdf, hcats, argss, uplots)

