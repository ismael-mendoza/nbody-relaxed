from src import utils
from src.frames import plots
from src.frames import catalogs
from src.frames.params import Param
from src import plot_funcs


def test_plot():
    cat_file = utils.data_path.joinpath('bolshoi/hlist_1.00035.csv')
    hcat = catalogs.HaloCatalog(cat_file, "Bolshoi", use_generator=False)

    params = [(Param('mvir', log=True), Param('T/|U|', log=True))]
    bplot = plots.BiPlot(plot_funcs.scatter_binning, params, nrows=1, ncols=1)
    bplot.run(hcat.cat, nxbins=10, xlabel_size=24, ylabel_size=24, tick_size=18, no_bars=True, show_bands=True,
              legend_label='all galaxies')
    bplot.save("test1.pdf")


# scatter_binning(np.log10(cat['mvir']), np.log10(cat['T/|U|']), nxbins=10, ax=axes[0],
#                xlabel=to_latex('mvir', True, True), ylabel=to_latex('T/|U|',True), xlabel_size=24, ylabel_size=24,
#                tick_size=18, no_bars=True, show_bands=True, legend_label='all galaxies')


