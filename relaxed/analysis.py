import json
from pathlib import Path

import numpy as np
from astropy.cosmology import LambdaCDM
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import findiff


from relaxed import halo_catalogs


def setup(
    name="m11",
    path="../../output",
    cutoff=0.1,
    particle_mass=1.35e8,
    particle_res=50,
    min_mass_bin=0.1,
):
    """ "Get catalog, indices, scales from given catalog pipeline output name.

    * cutoff is the percentage of haloes at a given scale that we tolerate with < 50 particles.

    """
    output = f"{path}/output_{name}/"
    cat_file = Path(output, "final_table.csv")
    z_map_file = Path(output, "z_map.json")

    # load all available scales and indices.
    with open(z_map_file, "r") as fp:
        scale_map = json.load(fp)  # map from i -> scale
    indices = np.array(list(scale_map.keys()))
    scales = np.array(list(scale_map.values()))

    # load catalog.
    hcat = halo_catalogs.HaloCatalog("Bolshoi", cat_file, label=name)
    hcat.load_cat_csv()
    avg_mass = np.nanmean(hcat.cat["mvir"])  # should be a narrow mass bina anyway.
    min_mass = particle_res * particle_mass
    avg_min_m = min_mass / avg_mass

    # extract MAH and determine scale we should cutoff.
    ma, keep_ma = get_ma(hcat.cat, indices)
    ma = ma[keep_ma]
    m_cutoff = np.nanquantile(ma, cutoff, axis=0)
    keep_cutoff = m_cutoff > avg_min_m  # over scales NOT data points.

    # filter scales and indices
    indices = indices[keep_cutoff]
    scales = scales[keep_cutoff]
    ma = ma.T[keep_cutoff].T

    # get am too
    am, mass_bins = get_am(ma, scales, min_mass=min_mass_bin)

    return hcat, ma, am, scales, indices, mass_bins


def get_ma(cat, indices):
    assert "mvir_a0" in cat.colnames
    assert "mvir_a160" in cat.colnames
    keep = []
    ma = np.zeros((len(cat), len(indices)))
    for i, k in enumerate(indices):
        k = int(k)
        colname = f"mvir_a{k}"

        # get mass fraction at this scale
        mvir = cat["mvir"]
        ms = cat[colname]
        ms = ms / mvir
        ma[:, i] = ms

    keep = np.ones(len(ma), dtype=bool)
    for i in range(len(ma)):
        keep[i] = ~np.any(np.isnan(np.log(ma[i, :])))
    assert np.sum(np.isnan(np.log(ma[keep]))) == 0

    return ma, keep


def get_am(ma, scales, min_mass=0.1):
    """
    1. Inversion is only a well-defined process for monotonic functions, and m(a) for an
    individual halo isn't necessarily monotonic. To solve this, the standard redefinition of a(m0)
    is that it's the first a where m(a) > m0. (This is, for example, how Rockstar defines halfmass
    scales.)

    2. Next, first pick your favorite set of mass bins that you'll evaluate it at. I think
    logarithmic bins spanning 0.01m(a=1) to 1m(a=1) is pretty reasonable, but you should probably
    choose this based on the mass ranges which are the most informative.

    3. Now, for each halo with masses m(a_i), measure M(a_i) = max_j{ m(a_j) | j <= i}.
    Remove (a_i, M(a_i)) pairs where M(a_i) = M(a_{i-1}), since this will mess up the inversion.

    4. Use scipy.interpolate.interp1d to create a function, f(m), which evaluates a(m).
    For stability, you'll want to run the interpolation on log(a_i) and log(M(a_i)), not a_i and M
    (a_i).

    5. Evaluate f(m) at the mass bins you decided that you liked in step 2. Now you can run your
    pipeline on this, just like you did for m(a).
    """

    # 1. + 2.
    mass_bins = np.linspace(np.log(min_mass), np.log(1.0), 100)

    # 3. NOTE: We invert max -> min because start with a = 1. Make function monotonic.
    Ma = np.zeros_like(ma)
    for i in range(len(ma)):
        _min = ma[i][0]
        for j in range(len(ma[i])):
            if ma[i][j] < _min:
                _min = ma[i][j]
            Ma[i][j] = _min

    # 4.
    fs = []
    for i in range(len(Ma)):
        pairs = [(scales[0], Ma[i][0])]
        count = 0
        for j in range(1, len(Ma[i])):
            # 3. keep only pairs that do NOT satisfy (a_{j-1}, Ma_{j-1}) = (a_j, Ma_j)
            if pairs[count][1] != Ma[i][j]:
                pairs.append((scales[j], Ma[i][j]))
                count += 1
        _scales = np.array([pair[0] for pair in pairs])
        _Mas = np.array([pair[1] for pair in pairs])
        fs.append(interp1d(np.log(_Mas), np.log(_scales), bounds_error=False, fill_value=np.nan))

    # 5.
    am = np.array([np.exp(f(mass_bins)) for f in fs])
    return am, np.exp(mass_bins)


def get_lam(am):
    # log(a(m)) whilst removing nan's if present.
    # the mask is applied to each of the extra arguments in `args` for convenience.
    lam = np.log(am)
    keep = np.ones(len(lam), dtype=bool)
    for i in range(len(lam)):
        keep[i] = ~np.any(np.isnan(lam[i, :]))
    assert np.sum(np.isnan(lam[keep])) == 0
    return lam, keep


def get_ma_corrs(cat, param, indices):
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


def get_am_corrs(pvalues, am, box_keep=None):
    if box_keep is None:
        box_keep = np.ones(am.shape[0])

    corrs = []
    n_mass_bins = am.shape[1]
    for k in range(n_mass_bins):
        corrs.append(stats.spearmanr(pvalues[box_keep], am[:, k][box_keep], nan_policy="omit")[0])
    return np.array(corrs)


def add_box_indices(cat, boxes=8, box_size=250):
    # box_size is in Mpc
    # create a new row add it to the catalogu for which box it is in.
    assert int(boxes ** (1.0 / 3)) == boxes ** (1.0 / 3)
    box_per_dim = int(boxes ** (1.0 / 3))
    divides = np.linspace(0, box_size, box_per_dim + 1)[1:-1]  # only use the middle.
    cat.add_column(np.zeros(len(cat)), name="ibox")
    for k, dim in enumerate(["x", "y", "z"]):
        for d in divides:
            cat["ibox"] += 2 ** k * (d < cat[dim])


def vol_jacknife_err(fn, cat, *args, mode="dict"):
    # assumes cat has had its box indices added with the function above.
    n_boxes = int(np.max(cat["ibox"]) + 1)
    values = []
    for b in range(n_boxes):
        box_keep = cat["ibox"] != b
        value = fn(*args, box_keep=box_keep)
        values.append(value)

    if mode == "dict":
        d = {}
        for val in values:
            for k, v in val.items():
                d[k] = d[k] + [v] if d.get(k, None) is not None else [v]

        d = {k: np.array(v) for k, v in d.items()}
        return {k: np.sqrt(v.var(axis=0) * (n_boxes - 1)) for k, v in d.items()}

    if mode == "array":
        values = np.array(values)
        return np.sqrt(values.var(axis=0) * (n_boxes - 1))

    raise NotImplementedError()


# create function that converts scale to fractional tdyn
def get_fractional_tdyn(scale, tdyn, sim_name="Bolshoi"):
    sim = halo_catalogs.sims[sim_name]

    # get cosmology based on given sim
    cosmo = LambdaCDM(H0=sim.h * 100, Ob0=sim.omega_b, Ode0=sim.omega_lambda, Om0=sim.omega_m)

    # tdyn in Gyrs
    z = (1 / scale) - 1
    return (cosmo.age(0).value - cosmo.age(z).value) / tdyn


def get_an_from_am(am, mass_bins, mrange=(0.498, 0.51)):
    # mrange should be a narrow mass range containing only 1 mass bin.
    # default is a_{n} = a_{1/2}
    idx = np.where((mrange[0] < mass_bins) & (mass_bins < mrange[1]))[0].item()
    return am[:, idx]


def get_quantiles(arr):
    return np.vectorize(lambda x: stats.percentileofscore(arr, x))(arr) / 100.0


def get_gradient(f, x, k=1, acc=2):
    """
    - f is an array with samples from a function with shape (n_samples, n_features)
    - x is the points where f is evaluated, x is assumed to be a uniform grid (linear spacing)
        Assumed to be the same for all n_samples.
    - k controls the step size.
    - acc is the accuracy at which to take the derivative.
    """

    assert f.shape[1] >= 2 * k
    coeff_table = findiff.coefficients(deriv=1, acc=acc)
    grad = []

    for i in range(f.shape[1]):
        if i - k * (acc // 2) < 0:
            mode = "forward"
            hx = x[i + 1] - x[i]  # accretion rate changes spacing a little bit at some scale.
        elif i - k * (acc // 2) >= 0 and i + k * (acc // 2) < f.shape[1]:
            mode = "center"
            hx = x[i + 1] - x[i]
        else:
            mode = "backward"
            hx = x[i] - x[i - 1]

        coeffs = coeff_table[mode]["coefficients"]  # coefficients are indepenent of step size.
        offsets = coeff_table[mode]["offsets"] * k + i

        assert np.all((offsets < f.shape[1]) & (offsets >= 0))

        deriv = np.sum(f[:, offsets] * coeffs.reshape(1, -1), axis=1) / (hx * k)
        grad.append(deriv.reshape(-1, 1))

    return np.hstack(grad)


def get_savgol_grads(scales, ma, k=5, deriv=1, n_samples=200):
    log_a = np.log(scales)
    log_ma = np.log(ma)
    assert np.sum(np.isnan(log_ma)) == 0
    f_log_ma = interp1d(log_a, log_ma, bounds_error=False, fill_value=np.nan)
    log_a_unif = np.linspace(log_a[0], log_a[-1], n_samples)
    log_ma_unif = f_log_ma(log_a_unif)
    d_log_a = abs(log_a_unif[-1] - log_a_unif[0]) / (len(log_a_unif) - 1)
    gamma_unif = savgol_filter(
        log_ma_unif, polyorder=4, window_length=k, deriv=deriv, delta=d_log_a
    )
    f_gamma = interp1d(log_a_unif, gamma_unif, bounds_error=False, fill_value=np.nan)
    gamma_a = f_gamma(log_a)
    return gamma_a


def get_tt_indices(n_points, test_ratio=0.2):
    test_size = int(np.ceil(test_ratio * n_points))
    test_idx = np.random.choice(range(n_points), replace=False, size=test_size)
    assert len(test_idx) == len(set(test_idx))
    train_idx = np.array(list(set(range(n_points)) - set(test_idx)))
    assert set(train_idx).intersection(set(test_idx)) == set()
    assert max(max(test_idx), max(train_idx)) == n_points - 1
    assert min(min(test_idx), min(train_idx)) == 0
    return train_idx, test_idx
