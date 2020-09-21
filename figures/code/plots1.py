import numpy as np
from relaxed import plots, plot_funcs
from relaxed.halo_parameters import get_hparam


def plot_mvir_histogram(hcats, pdf):
    """Basic plot showcasing features with a single histogram of Mvir, multiple catalogs may be
    used at once.

    Args:
        hcats (list): Already loaded, list of hcat objects.
        pdf: PDF object that can be used to save figures to.
    """
    names = [hcat.name for hcat in hcats]

    # (1) Start with plot_func creation, just using default values for everything.
    create_histogram = plot_funcs.CreateHistogram(xlabel_size=24)

    # (2) Create list of all unique halo_params that are necessary for plotting.
    hparams = [get_hparam("mvir", log=True)]

    # (2) Create Plot object.
    histogram_plot = plots.Histogram(create_histogram, hparams, nrows=1, ncols=1)

    # (3) Load corresponding values of hparams from hcats into plot object
    for hcat in hcats:
        histogram_plot.load(hcat)

    # (4) Create a list of names of which order you want to plot your hparams
    plot_params = {"mvir": {*names}}

    # (4) Generate figure
    histogram_plot.generate(plot_params)
    histogram_plot.save(pdf=pdf)


def plot_with_mass1(hcats, pdf, colors):
    """
    Catalogs should already be loaded properly.Obtain some of the basic plots where multiple
    catalogs results might be overlaid. This includes:

    * Histograms of Mvir
    * Mean-centered histograms of relevant quantities.
    * Plots demonstrating the correlations between different relaxedness parameters and mvir.

    :param hcats: Already load, list of hcat objects.
    :param colors:
    :param pdf: already prepared pdf object to save figures into.
    :return: None
    """

    # Plot 1: histogram of Mvir

    # (2) Now for the plot.

    # this are the default values that we will be using throughout the plots.

    bin_bds = np.arange(11, 14.5, 0.5)
    binning_kwargs = dict(bin_bds=bin_bds, show_bands=True, **general_kwargs)

    # (1) Need to create all the plots and specify their parameters in kwargss.

    # Plot 1: histogram of Mvir
    params = [HaloParam("mvir", log=True)]
    plot1 = plots.Histogram(
        plot_funcs.histogram, params, nrows=1, ncols=1, plot_kwargs=hist_kwargs
    )

    # Plot 2: Relaxedness parameters and mvir
    relaxed_params = [
        HaloParam("eta", log=True),
        HaloParam("x0", log=True),
        HaloParam("v0", log=True),
        HaloParam("xoff", log=True),
        HaloParam("voff", log=True),
        HaloParam("q", log=True),
        HaloParam("cvir", log=True),
        HaloParam("f_sub", log=False),
        HaloParam("a2", log=True),
    ]
    params = [(HaloParam("mvir", log=True), param) for param in relaxed_params]

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


#
# def plot_mean_centered_hists(hcats, pdf, colors):
#     general_kwargs = dict(xlabel_size=28, ylabel_size=28)
#
#     hist_kwargs2 = dict(
#         bins=30,
#         histtype="step",
#         extra_hist_kwargs=dict(),
#         log_y=True,
#         vline="median",
#         **general_kwargs
#     )
#
#     # Plot 2: mean-centered histogram of relevant quantities
#     # title = "Mean centered histograms"
#     modifiers = [lambda x: (x - np.mean(x)) / np.std(x)]
#     params = [
#         HaloParam("cvir", log=True, modifiers=modifiers),
#         HaloParam("eta", log=True, modifiers=modifiers),
#         HaloParam("x0", log=True, modifiers=modifiers),
#         HaloParam("v0", log=True, modifiers=modifiers),
#         HaloParam("spin", log=True, modifiers=modifiers),
#         HaloParam("q", log=True, modifiers=modifiers),
#         HaloParam("phi_l", log=True, modifiers=modifiers),
#         HaloParam("f_sub", log=False),
#     ]
#     plot2 = plots.Histogram(
#         plot_funcs.histogram,
#         params,
#         ncols=2,
#         nrows=4,
#         figsize=(12, 20),
#         title=None,
#         title_size=24,
#         plot_kwargs=hist_kwargs2,
#     )
#
#     uplots = [plot2]
#     hplots = [[uplot for uplot in uplots] for _ in hcats]
#     generate_and_save(pdf, hcats, hplots, uplots, colors=colors, cached=False)
#
#
# def plot_correlation_matrix_basic(hcats, pdf=None):
#     """
#     Create a visualization fo the matrix of correlation separate for each of the catalogs in hcats.
#     :param hcats:
#     :param pdf:
#     :return:
#     """
#
#     hplots = [[] for _ in hcats]
#     uplots = []
#
#     # Plot 4: Matrix correlations
#     params = [
#         HaloParam("mvir", log=True),
#         HaloParam("cvir", log=True),
#         HaloParam("eta", log=True),
#         HaloParam("x0", log=True),
#         HaloParam("v0", log=True),
#         HaloParam("spin", log=True),
#         HaloParam("q", log=True),
#         HaloParam("f_sub", log=False),
#         HaloParam("a2", log=True),
#         HaloParam("phi_l", log=True),
#     ]
#     for hcat, hplot in zip(hcats, hplots):
#         kwargs = dict(label_size=10, show_cell_text=True)
#         plot = plots.MatrixPlot(
#             stats.get_corrs,
#             params,
#             symmetric=False,
#             plot_kwargs=kwargs,
#             title=hcat.label,
#             title_size=24,
#         )
#         uplots.append(plot)
#         hplot.append(plot)
#
#     generate_and_save(pdf, hcats, hplots, uplots)
#
#
# def plot_decades_basic(hcats, pdf, colors):
#     """Produce all the basic plots that require decade separation"""
#
#     general_kwargs = dict(xlabel_size=28, ylabel_size=28)
#     binning_kwargs = dict(n_xbins=8, show_bands=False, **general_kwargs)
#
#     figsize = (24, 24)
#     uplots = []
#     hplots = [[] for _ in hcats]
#
#     # Plot 5: Correlation between all pairs of different relaxedness parameters as a function
#     # of mass half decades.
#     # params to include:  't/|u|', 'x0', 'v0', 'xoff', 'Voff', 'q', 'cvir'
#     params = [
#         (HaloParam("t/|u|", log=True), HaloParam("x0", log=True)),
#         (HaloParam("t/|u|", log=True), HaloParam("v0", log=True)),
#         (HaloParam("t/|u|", log=True), HaloParam("q", log=True)),
#         (HaloParam("t/|u|", log=True), HaloParam("cvir", log=True)),
#         (HaloParam("x0", log=True), HaloParam("v0", log=True)),
#         (HaloParam("x0", log=True), HaloParam("q", log=True)),
#         (HaloParam("x0", log=True), HaloParam("cvir", log=True)),
#         (HaloParam("v0", log=True), HaloParam("q", log=True)),
#         (HaloParam("v0", log=True), HaloParam("cvir", log=True)),
#         (HaloParam("q", log=True), HaloParam("cvir", log=True)),
#     ]  # total = 10
#     param_locs = []  # triangular pattern, user defined param_locs.
#     for i in range(4):
#         for j in range(4 - i):
#             param_locs.append((j, i))
#
#     plot1 = plots.BiPlot(
#         plot_funcs.scatter_binning,
#         params,
#         nrows=4,
#         ncols=4,
#         figsize=figsize,
#         param_locs=param_locs,
#         plot_kwargs=binning_kwargs,
#         title_size=40,
#     )
#
#     uplots.append(plot1)
#     for hplot in hplots:
#         hplot.append(plot1)
#
#     generate_and_save(pdf, hcats, hplots, uplots, colors=colors, cached=False)
