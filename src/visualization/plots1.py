from src.utils import const
from src.frames import plots
from src.frames import catalogs
from src.frames.params import Param
from src import plot_funcs
import matplotlib.backends.backend_pdf as back_pdf


def test_plot(hcat=None, cat_filename=None, catalog_name="Bolshoi"):
    assert hcat is not None or cat_filename is not None, "Need at least one"

    # load catalog
    if hcat is None:
        cat_file = const.data_path.joinpath(cat_filename)
        hcat = catalogs.HaloCatalog(cat_file, catalog_name, use_generator=False, subhalos=False)
        relaxed_hcat = catalogs.HaloCatalog(cat_file, catalog_name, use_generator=False, subhalos=False, relaxed=True)

    default_hist_kwargs = dict(bins=30, histtype='step')
    default_kwargs1 = dict(nxbins=10, xlabel_size=24, ylabel_size=24,
                           tick_size=18, no_bars=True, show_bands=True)

    # what params to use
    params = [(Param('mvir', log=True), Param('T/|U|', log=True))]
    bplot1 = plots.BiPlot(plot_funcs.scatter_binning, params, nrows=1, ncols=1)
    bplot1.run(hcat.cat, **default_params1)

    params = [(Param('mvir', log=True), Param('Xoff', log=True))]
    bplot2 = plots.BiPlot(plot_funcs.scatter_binning, params, nrows=1, ncols=1)
    bplot2.run(hcat.cat, **default_params1)

    # prepare pdf to save to.
    pdf = back_pdf.PdfPages(const.figure_path.joinpath("test_bolshoi.pdf"))
    bplot1.save(pdf=pdf)
    bplot2.save(pdf=pdf)
    pdf.close()


# scatter_binning(np.log10(cat['mvir']), np.log10(cat['T/|U|']), nxbins=10, ax=axes[0],
#                xlabel=to_latex('mvir', True, True), ylabel=to_latex('T/|U|',True), xlabel_size=24, ylabel_size=24,
#                tick_size=18, no_bars=True, show_bands=True, legend_label='all galaxies')


