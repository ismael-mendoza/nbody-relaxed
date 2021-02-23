import json
from pathlib import Path

import numpy as np
from astropy.cosmology import LambdaCDM
from scipy import stats
from scipy.interpolate import interp1d

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
    # we are removing from the end bc that's how scales are ordered.
    scales = scales[keep]

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


def get_ma(cat, indices):
    assert "mvir_a0" in cat.colnames
    assert "mvir_a160" in cat.colnames
    ma = np.zeros((len(cat), len(indices)))
    for k in indices:
        k = int(k)
        colname = f"mvir_a{k}"

        # get mass fraction at this scale
        mvir = cat["mvir"]
        ms = cat[colname]
        ms = ms / mvir
        ma[:, k] = ms

    return ma


def get_am(name="m11"):
    """
    Here are the steps that Phil outlined (in slack) to do this:

    1. Inversion is only a well-defined process for monotonic functions, and m(a) for an individual halo isn't necessarily monotonic. To solve this, the standard redefinition of a(m0) is that it's the first a where m(a) > m0. (This is, for example, how Rockstar defines halfmass scales.)

    2. Next, first pick your favorite set of mass bins that you'll evaluate it at. I think logarithmic bins spanning 0.01m(a=1) to 1m(a=1) is pretty reasonable, but you should probably choose this based on the mass ranges which are the most informative once you.

    3. Now, for each halo with masses m(a_i), measure M(a_i) = max_j{ m(a_j) | j <= i}.
    Remove (a_i, M(a_i)) pairs where M(a_i) = M(a_{i-1}), since this will mess up the inversion.

    4. Use scipy.interpolate.interp1d to create a function, f(m), which evaluates a(m).
    For stability, you'll want to run the interpolation on log(a_i) and log(M(a_i)), not a_i and M(a_i).

    5. Evaluate f(m) at the mass bins you decided that you liked in step 2. Now you can run your pipeline on this, just like you did for m(a).
    """
    hcat, indices, scales = setup(name)

    # 2.
    mass_bins = np.linspace(np.log(0.01), np.log(1.0), 100)

    # 3.
    ma = get_ma(hcat.cat, indices)
    Ma = np.zeros_like(ma)
    for i in range(len(ma)):
        _min = ma[i][0]
        for j in range(len(ma[i])):
            if ma[i][j] < _min:
                _min = ma[i][j]
            Ma[i][j] = _min

    # 4. + 5.
    # We will get the interpolation for each halo separately
    fs = []
    for i in range(len(Ma)):
        pairs = [(scales[0], Ma[i][0])]
        count = 0
        for j in range(1, len(Ma[i])):
            # keep only pairs that do NOT satisfy (a_{j-1}, Ma_{j-1}) = (a_j, Ma_j)
            if pairs[count][1] != Ma[i][j]:
                pairs.append((scales[j], Ma[i][j]))
                count += 1
        _scales = np.array([pair[0] for pair in pairs])
        _Mas = np.array([pair[1] for pair in pairs])
        fs.append(
            interp1d(
                np.log(_Mas), np.log(_scales), bounds_error=False, fill_value=np.nan
            )
        )

    # 6.
    am = np.array([np.exp(f(mass_bins)) for f in fs])
    return am, np.exp(mass_bins)
