import json
from pathlib import Path

import findiff
import numpy as np
from astropy.cosmology import LambdaCDM
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from relaxed import halo_catalogs


def determine_cutoffs(
    Mvir,
    scales,
    cutoff_missing=0.01,
    cutoff_particle=0.1,
    particle_mass=1.35e8,
    particle_res=50,
):
    """Return minimum scale to use for m(a) and minimum mass bin to use for a(m).

    Args:
        * Mvir: Raw virial mass accretion history, array w/ shape (n_samples, n_scales).

        * cutoff_particle: is the percentage of haloes at a given scale that we tolerate
            with < `particle_res` particles,

        * cutoff_missing: the percentage of haloes with no progenitors we are OK discarding.
            NOTE: Phil suggested to make this small so that we don't bias our samples
            (late-forming haloes are more likely to have nan's at the end.)

    """
    # minimum virial mass we are comfortable with
    min_mass = particle_res * particle_mass

    # NOTE: Assumes that order is early -> late.
    # determine scales we should cutoff based on percentage of NaN progenitors.
    n_nan = np.sum(np.isnan(Mvir), axis=0) / len(Mvir)
    idx1 = np.where(n_nan > cutoff_missing)[0][-1] + 1
    min_scale1 = scales[idx1]

    # NOTE: Assumes that order is early -> late.
    # NOTE: These two cuts are different because ROCKSTAR does not limit to ~50 particles.
    # determine scale we should cutoff based on num. of particles.
    m_cutoff = np.nanquantile(Mvir, cutoff_particle, axis=0)  # it is monotonically decreasing
    idx2 = np.where(m_cutoff < min_mass)[0][-1] + 1  # over scales NOT data points.
    min_scale2 = scales[idx2]

    # combine two criteria into one scale
    min_scale = max(min_scale1, min_scale2)

    # Now we want to do the same but for mass bins.
    # NOTE: The explanation is that we want (1) minimum mass bin m0 such that at least
    # 99% haloes have a progenitor at that mass bin and (2) minimum mass bin m0 such that
    # 90% of haloes have their earliest a0 s.t. m(a0) > m0 satisfy m(a0) * Mvir(a=1)/ 1.35e8 > 50
    min_mass_bin1 = np.nanquantile(np.nanmin(Mvir, axis=1) / Mvir[:, -1], cutoff_missing)
    min_mass_bin2 = np.nanquantile(min_mass / Mvir[:, -1], 1 - cutoff_particle)
    assert isinstance(min_mass_bin1, float)
    assert isinstance(min_mass_bin2, float)
    min_mass_bin = max(min_mass_bin1, min_mass_bin2)

    return min_scale, min_mass_bin


def get_mah(
    name="m11",
    path="../../output",
    cutoff_missing=0.01,
    cutoff_particle=0.1,
    particle_mass=1.35e8,  # Bolshoi
    particle_res=50,
    n_bins=100,
    m_version="vir",
):
    """Get catalog, indices, scales from given catalog pipeline output name."""
    output = f"{path}/output_{name}/"
    cat_file = Path(output, "final_table.csv")
    z_map_file = Path(output, "z_map.json")

    # load all available scales and indices.
    with open(z_map_file, "r") as fp:
        scale_map = json.load(fp)  # map from i -> scale
    indices = np.array(list(scale_map.keys()))
    scales = np.array(list(scale_map.values()))
    sort_idx = np.argsort(scales)  # want order: early -> late. Low scales -> high scales (1.0)
    indices = indices[sort_idx].astype(int)  # easier to handle.
    scales = scales[sort_idx]

    # load catalog.
    hcat = halo_catalogs.HaloCatalog("Bolshoi", cat_file, label=name)
    hcat.load_cat_csv()
    cat = hcat.cat

    # extract m(a) information.
    ma_info = get_ma_info(cat, indices)
    ma = ma_info[f"ma_{m_version}"]  # no cut yet.

    # get relevant cutoff on scales and mass bins (defined on m(a)).
    min_scale, min_mass_bin = determine_cutoffs(
        ma_info["Mvir"], scales, cutoff_missing, cutoff_particle, particle_mass, particle_res
    )

    # fill nan's with average `m` value of 1 particle
    avg_mass = np.nanmean(cat["mvir"])
    ma[np.isnan(ma)] = particle_mass / avg_mass

    # obtain a(m) and corresponding mass_bins
    am, mass_bins = get_am(ma, scales, min_mass_bin, n_bins)

    # filter scales, indices, m(a) based on `min_scale`.
    keep_scale = scales > min_scale
    indices = indices[keep_scale]
    scales = scales[keep_scale]
    ma = ma.T[keep_scale].T

    assert np.sum(np.isnan(ma)) == 0

    # FIXME: Need to filter some f_subs using a similar criteria as with m(a), a(m)
    # and f_sub(a)
    # f_sub = np.array([cat[f"f_sub_a{i}"].data for i in indices]).T

    # remove haloes with np.nan in a(m)
    keep_am = ~np.isnan(np.sum(am, axis=1))
    cat = cat[keep_am]
    ma = ma[keep_am]
    am = am[keep_am]

    return {
        "cat": cat,
        "ma": ma,
        "am": am,
        "scales": scales,
        "indices": indices,
        "mass_bins": mass_bins,
    }


def get_ma_info(cat, indices):
    assert "mvir_a0" in cat.colnames
    assert "mvir_a160" in cat.colnames
    Mvir = np.zeros((len(cat), len(indices)))
    for i, idx in enumerate(indices):
        colname = f"mvir_a{idx}"
        Mvir[:, i] = cat[colname]
    Mpeak = np.fmax.accumulate(Mvir, axis=1)  # fmax ignores nan's, but keeps the beginning ones.

    ma_vir = Mvir / Mvir[:, -1].reshape(-1, 1)
    ma_peak = Mpeak / Mpeak[:, -1].reshape(-1, 1)
    ma_mix = Mvir / Mpeak[:, -1].reshape(-1, 1)

    return {"ma_vir": ma_vir, "ma_peak": ma_peak, "ma_mix": ma_mix, "Mvir": Mvir, "Mpeak": Mpeak}


def get_am(ma, scales, min_mass_bin, n_bins=100):
    """
    1. Inversion is only a well-defined process for monotonic functions, and m(a) for an
    individual halo isn't necessarily monotonic. To solve this, the standard redefinition of a(m0)
    is that it's the first a where m(a) > m0. (This is, for example, how Rockstar defines halfmass
    scales.)

    2. Next, first pick your favorite set of mass bins that you'll evaluate it at. I think
    logarithmic bins spanning 0.01m(a=1) to 1m(a=1) is pretty reasonable, but you should probably
    choose this based on the mass ranges which are the most informative.

    3. Now, for each halo with masses m(a_i), measure M_peak(a_i) = max_j{ m(a_j) | j <= i}.

    4. Remove (a_i, M_peak(a_i)) pairs where M_peak(a_i) = M_peak(a_{i-1}), since this will mess
    up the inversion.

    5. Use scipy.interpolate.interp1d to create a function, f(m), which evaluates a(m).
    For stability, you'll want to run the interpolation on log(a_i) and log(M(a_i)), not a_i and M
    (a_i).

    6. Evaluate f(m) at the mass bins you decided that you liked in step 2. Now you can run your
    pipeline on this, just like you did for m(a).
    """
    assert np.sum(np.isnan(ma)) == 0, "m(a) needs to be filled with `fill_value` previously."

    # 1. + 2.
    mass_bins = np.linspace(np.log(min_mass_bin), np.log(1.0), n_bins)

    # 3. NOTE: Make function monotonic. We assume start is early -> late ordering.
    # NOTE: ma should not have any cuts.
    m_peak = np.fmax.accumulate(ma, axis=1)  # fmax ignores nan's (except beginning ones)

    # 4. + 5.
    fs = []
    for i in range(len(m_peak)):
        pairs = [(scales[0], m_peak[i][0])]
        count = 0
        for j in range(1, len(m_peak[i])):
            # keep only pairs that do NOT satisfy (a_{j-1}, Ma_{j-1}) = (a_j, Ma_j)
            if pairs[count][1] != m_peak[i][j]:
                pairs.append((scales[j], m_peak[i][j]))
                count += 1

        if len(pairs) == 1:
            raise AssertionError(
                "Only 1 pair was added, so max was reached at a -> 0 which is impossible."
            )

        _scales = np.array([pair[0] for pair in pairs])
        _m_peaks = np.array([pair[1] for pair in pairs])
        fs.append(
            interp1d(np.log(_m_peaks), np.log(_scales), bounds_error=False, fill_value=np.nan)
        )

    # 6.
    am = np.array([np.exp(f(mass_bins)) for f in fs])
    return am, np.exp(mass_bins)


def get_ma_corrs(cat, param, indices):
    corrs = []
    for k in indices:
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


def get_an_from_am(am, mass_bins, mbin=0.498):
    # `mbin` is the bin you want to get from am, returns first bin bigger than `mbin`
    # default is `a_{n} = a_{1/2}`
    idx = np.where(mass_bins > mbin)[0][0]
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
