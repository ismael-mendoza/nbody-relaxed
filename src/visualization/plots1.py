from src.utils import const
from src.frames import plots
from src.frames.catalogs import HaloCatalog
from src.frames.params import Param
from src import plot_funcs
from src import stats
import matplotlib.backends.backend_pdf as back_pdf

import numpy as np


def test_plot(hcat=None, cat_filename=None, catalog_name="Bolshoi"):
    assert hcat is not None or cat_filename is not None, "Need at least one of hcat or cat_filename"

    # this are the default values that we will be using throughout the plots.
    hist_kwargs1 = dict(bins=30, histtype='step')
    binning_kwargs1 = dict(nxbins=10, no_bars=True, show_bands=True)
    binning_3d_kwargs1 = dict(nxbins=10, no_bars=False, show_bands=False)
    general_kwargs1 = dict(xlabel_size=24, ylabel_size=24)

    # (1) create all the necessary catalogs.
    cat_file = const.data_path.joinpath(cat_filename)
    hcat1 = HaloCatalog(cat_file, catalog_name, subhalos=False)
    hcat1.set_cat(use_generator=False)

    relaxed_hcat = HaloCatalog.create_relaxed_from_complete(hcat1, relaxed='power2011')
    hcats = [hcat1, relaxed_hcat]
    colors = ['red', 'blue']

    # (2) Need to create all the plots and specify their parameters in kwargss.
    all_plots = []
    kwargss = []

    # Plot 1: histogram of Mvir
    params = [Param('mvir', log=True)]
    all_plots.append(
        plots.UniPlot(plot_funcs.histogram, params, nrows=1, ncols=1)
    )
    kwargss.append(hist_kwargs1)

    # Plot 2: mean-centered histogram of relevant quantities
    modifiers = [lambda x: (x - np.mean(x)) / np.std(x)]
    param_names = ['mvir', 'cvir', 'T/|U|', 'xoff', 'voff', 'Spin', 'q', 'phi_l']
    params = [Param(param_name, log=True, modifiers=modifiers) for param_name in param_names]
    all_plots.append(
        plots.UniPlot(plot_funcs.histogram, params, ncols=2, nrows=4)
    )
    kwargss.append({'hist_kwargs': hist_kwargs1})

    # Plot 3: Matrix correlations
    param_names = ['mvir', 'cvir', 'T/|U|', 'xoff', 'voff', 'Spin', 'q', 'phi_l']
    params = [Param(param_name, log=True) for param_name in param_names]
    all_plots.append(
        plots.MatrixPlot(stats.get_corrs, params, symmetric=False)
    )
    kwargss.append({})

    # Plot 4: Relaxedness parameters and mvir
    relaxedness_param_names = ['T/|U|', 'xoff', 'voff', 'Xoff', 'Voff', 'q', 'cvir']
    params = [(Param('mvir', log=True), Param(relaxed_param, log=True)) for relaxed_param in relaxedness_param_names]
    all_plots.append(
        plots.BiPlot(plot_funcs.scatter_binning, params, nrows=3, ncols=2)
    )
    kwargss.append(binning_kwargs1)

    # Plot 5: Correlation between the different relaxedness parameters as a function of mass half decades.
    # ToDo: adapt it so this can be plotted here too.
    relaxedness_param_names = ['T/|U|', 'xoff', 'voff', 'Xoff', 'Voff', 'q', 'cvir']
    params = []
    for param_name1 in relaxedness_param_names:
        for param_name2 in relaxedness_param_names:
            params.append(
                        (Param(param_name1, log=True), Param(param_name2, log=True))
                        )
    all_plots.append(
        plots.BiPlot(plot_funcs.binning3d_mass, params, nrows=3, ncols=7)
    )
    kwargss.append(binning_3d_kwargs1)

    # Plot 6: Same plot as above but without any decades.

    # (3) Prepare pdf file to save the figures to.
    pdf = back_pdf.PdfPages(const.figure_path.joinpath("test_bolshoi.pdf"))

    # (4) Now, for each of the catalogs we actually create the plots and save them.
    for hcat, color in zip(hcats, colors):
        for plot, kwargs in zip(all_plots, kwargss):
            plot.generate(hcat.cat, legend_label=hcat.catalog_label, color=color, **kwargs, **general_kwargs1)
            plot.save(pdf=pdf)

    # (5) Close the pdf.
    pdf.close()
