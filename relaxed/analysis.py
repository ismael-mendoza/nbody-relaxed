import json
from pathlib import Path

import numpy as np
from astropy.cosmology import LambdaCDM
from scipy import stats

from . import halo_catalogs


def setup(name="m11", path="../../temp"):
    # get catalog, indices, and scales (redshift) from given catalog pipeline output name
    output = f"{path}/output_{name}/"
    cat_file = Path(output, "final_table.csv")
    z_map_file = Path(output, "z_map.json")

    with open(z_map_file, "r") as fp:
        scale_map = json.load(fp)  # map from i -> scale

    # only keep stable scales.
    indices = np.array(list(scale_map.keys()))
    scales = np.array(list(scale_map.values()))
    keep = scales > 0.15
    indices = indices[keep]
    scales = scales[
        keep
    ]  # we are removing from the end bc that's how scales are ordered.

    # load catalog.
    hcat = halo_catalogs.HaloCatalog("Bolshoi", cat_file, label=name)
    hcat.load_cat_csv()

    # remove weird ID
    hcat.cat = hcat.cat[hcat.cat["mvir_a18"] > 0]

    return hcat, indices, scales


def get_m_a_corrs(cat, param, indices):
    corrs = []
    for k in indices:
        k = int(k)
        colname = f"mvir_a{k}"
        keep = (~np.isnan(cat[colname])) & (cat[colname] > 0)

        # get mass fraction at this scale
        mvir = cat["mvir"][keep]
        ms = cat[colname][keep]
        ms = ms / mvir
        pvalue = cat[param][keep]

        # get correlation.
        assert np.all(ms > 0) and np.all(~np.isnan(ms))
        assert np.all(mvir > 0)
        corr = stats.spearmanr(ms, pvalue)[0]
        corrs.append(corr)

    return np.array(corrs)


def get_am_corrs(cat, param, am):
    corrs = []
    n_mass_bins = am.shape[1]
    for k in range(n_mass_bins):
        keep = ~np.isnan(am[:, k])

        # get mass fraction at this scale
        am_k = am[:, k][keep]
        pvalue = cat[param][keep]

        # get correlation.
        assert np.all(~np.isnan(pvalue)) and np.all(~np.isnan(am_k))
        corr = stats.spearmanr(pvalue, am_k)[0]
        corrs.append(corr)

    return np.array(corrs)


def add_box_indices(cat, boxes=8, box_size=250):
    # create a new row add it to the catalogu for which box it is in.
    assert int(boxes ** (1.0 / 3)) == boxes ** (1.0 / 3)
    box_per_dim = int(boxes ** (1.0 / 3))
    divides = np.linspace(0, box_size, box_per_dim + 1)[1:-1]  # only use the middle.
    cat.add_column(np.zeros(len(cat)), name="ibox")
    for k, dim in enumerate(["x", "y", "z"]):
        for d in divides:
            cat["ibox"] += 2 ** k * (d < cat[dim])


def vol_jacknife_values(f, cat, param, *args):
    # assumes cat has had its box indices added with the function above.
    n_boxes = int(np.max(cat["ibox"]) + 1)
    values = []
    for b in range(n_boxes):
        _cat = cat[cat["ibox"] != b]
        value = f(_cat, param, *args)
        values.append(value)
    return np.array(values)


# create function that converts scale to fractional tdyn
def get_fractional_tdyn(scale, tdyn, sim_name="Bolshoi"):
    sim = halo_catalogs.sims[sim_name]

    # get cosmology based on given sim
    cosmo = LambdaCDM(
        H0=sim.h * 100, Ob0=sim.omega_b, Ode0=sim.omega_lambda, Om0=sim.omega_m
    )

    # tdyn in Gyrs
    z = (1 / scale) - 1
    return (cosmo.age(0).value - cosmo.age(z).value) / tdyn
