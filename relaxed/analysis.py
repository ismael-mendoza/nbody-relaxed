import json
from pathlib import Path
from relaxed.halo_parameters import X

import numpy as np
from astropy.cosmology import LambdaCDM
from scipy import stats
from scipy.interpolate import interp1d

from relaxed import halo_catalogs


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

    return hcat, indices, scales


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


def get_am(name="m11", min_mass=0.1, path="../../temp"):
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
    hcat, indices, scales = setup(name, path=path)

    # 2.
    mass_bins = np.linspace(np.log(min_mass), np.log(1.0), 100)

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
        fs.append(interp1d(np.log(_Mas), np.log(_scales), bounds_error=False, fill_value=np.nan))

    # 6.
    am = np.array([np.exp(f(mass_bins)) for f in fs])
    return am, np.exp(mass_bins)


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


def get_am_corrs(cat, param, am):
    corrs = []
    n_mass_bins = am.shape[1]
    for k in range(n_mass_bins):
        corrs.append(stats.spearmanr(cat[param], am[:, k], nan_policy="omit")[0])
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
    cosmo = LambdaCDM(H0=sim.h * 100, Ob0=sim.omega_b, Ode0=sim.omega_lambda, Om0=sim.omega_m)

    # tdyn in Gyrs
    z = (1 / scale) - 1
    return (cosmo.age(0).value - cosmo.age(z).value) / tdyn


def get_a2_from_am(am, mass_bins):
    idx = np.where((0.498 < mass_bins) & (mass_bins < 0.51))[0].item()
    return am[:, idx]


def get_lam(am, *args):
    # log(a(m)) whilst removing nan's if present.
    # the mask is applied to each of the extra arguments in `args` for convenience.
    lam = np.log(am)
    keep = np.ones(len(lam), dtype=bool)
    for i in range(len(lam)):
        keep[i] = ~np.any(np.isnan(lam[i, :]))

    assert np.sum(np.isnan(lam[keep])) == 0
    return keep, lam[keep], *(arg[keep] for arg in args)


def get_quantiles(arr):
    return np.vectorize(lambda x: stats.percentileofscore(arr, x))(arr) / 100.0


def gaussian_conditional(x, y):
    # y (usually) represents one of the dark matter halo properties at z=0.
    # x are the features used for prediction, should have shape (y.shape[0], n_features)

    assert len(y.shape) == 1 and len(x.shape) == 2
    assert x.shape[0] == y.shape[0]
    n_features = x.shape[1]

    # calculate sigma/correlation matrix bewteen all quantities
    z = np.vstack([y.reshape(1, -1), x.T]).T

    # some sanity checks
    assert z.shape == (y.shape[0], n_features + 1)
    np.testing.assert_equal(y, z[:, 0])
    np.testing.assert_equal(x[:, 0], z[:, 1])  # ignore mutual nan's
    np.testing.assert_equal(x[:, -1], z[:, -1])

    # calculate covariances
    Sigma = np.zeros((1 + n_features, 1 + n_features))
    rho = np.zeros((1 + n_features, 1 + n_features))
    for i in range(n_features + 1):
        for j in range(n_features + 1):
            if i <= j:
                # calculate correlation coefficient keepin only non-nan values
                z1, z2 = z[:, i], z[:, j]
                keep = ~np.isnan(z1) & ~np.isnan(z2)
                cov = np.cov(z1[keep], z2[keep])
                assert cov.shape == (2, 2)
                Sigma[i, j] = cov[0, 1]
                rho[i, j] = np.corrcoef(z1[keep], z2[keep])[0, 1]
            else:
                rho[i, j] = rho[j, i]
                Sigma[i, j] = Sigma[j, i]

    # more sanity
    assert np.all(~np.isnan(Sigma))
    assert np.all(~np.isnan(rho))

    # we assume a multivariate-gaussian distribution P(X, a(m1), a(m2), ...) with
    # conditional distribution P(X | {a(m_i)}) uses the rule here:
    # https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
    # we return the mean/std deviation of the conditional gaussian.
    mu1 = np.nanmean(y).reshape(1, 1)
    mu2 = np.nanmean(x, axis=0).reshape(n_features, 1)
    Sigma11 = Sigma[0, 0].reshape(1, 1)
    Sigma12 = Sigma[0, 1:].reshape(1, n_features)
    Sigma22 = Sigma[1:, 1:].reshape(n_features, n_features)

    def mu_cond(x_test):
        # computes mu(y | X=x_test)
        assert np.sum(np.isnan(x_test)) == 0
        x_test = x_test.reshape(-1, n_features).T
        mu_cond = mu1 + Sigma12.dot(np.linalg.inv(Sigma22)).dot(x_test - mu2)
        return mu_cond.reshape(-1)

    sigma_cond = Sigma11 - Sigma12.dot(np.linalg.inv(Sigma22)).dot(Sigma12.T)
    return {
        "mu1": mu1,
        "mu2": mu2,
        "Sigma": Sigma,
        "rho": rho,
        "mu_cond": mu_cond,
        "sigma_cond": sigma_cond,
    }


def training_suite(x, y, suite=("LN-RS",), extra_args: dict = None):
    """
    Args:
        x (np.array): (Unlogged) Training data array with features and shape (n_points, n_features)
        y (np.array): (Unlogged) Training data array with variable to be predicted
            (e.g. cvir, xoff, or eta)

    Legend:
        - LN-RS: LogNormal random samples.
        - MG-LFC: Multi-Variate Gaussian using full covariance matrix. (returns conditional mean)
            with logged variables.
        - CAM: Conditional Abundance Matching
        - MG-A2: Bivariate Gaussian only usig a_{1/2} for conditional estimate.
        - MV-LR: Linear regression with no logs.
        - MV-LLR: Multi-Variate Linear Regression with logs
        - MG-TFC: Multi-Gaussian after transforming (non-log) variables with quantile transformer.
        - MV-TLR: Linear regression after transforming variables with a quantile transformer.
    """
    assert set(suite).issubset(
        {"MG-LFC", "LN-RS", "CAM", "MG-A2", "MV-LLR", "MV-LR", "MG-TFC", "MV-TLR"}
    )

    assert np.isnan(np.log(x)).sum() == 0
    assert np.isnan(np.log(y)).sum() == 0

    trained_models = {}

    # whether using suites that require using a gaussian remapping transformation.
    if "MV-TLR" in suite or "MG-TFC" in suite:
        from sklearn.preprocessing import QuantileTransformer

        qt_y = QuantileTransformer(n_quantiles=len(y), output_distribution="normal").fit(
            y.reshape(-1, 1)
        )
        qt_x = QuantileTransformer(n_quantiles=len(x), output_distribution="normal").fit(x)

        y_trans = qt_y.transform(y.reshape(-1, 1)).reshape(-1)
        x_trans = qt_x.transform(x)

    if "LN-RS" in suite:
        mu, sigma = np.mean(np.log(y)), np.std(np.log(y))

        def lognormal(x_test):
            n_test = len(x_test)
            log_Y_pred = np.random.normal(mu, sigma, n_test)
            return np.exp(log_Y_pred)

        trained_models["LN-RS"] = lognormal

    if "MG-LFC" in suite:

        # multivariate prediction
        gcond = gaussian_conditional(np.log(x), np.log(y))

        def multi_gauss(x_test):
            return np.exp(gcond["mu_cond"](np.log(x_test)))

        trained_models["MG-LFC"] = multi_gauss

    if "MG-TFC" in suite:
        gcond = gaussian_conditional(x_trans, y_trans)

        def multi_gauss_trans(x_test):
            x_trans_test = qt_am.transform(x_test)
            y_trans_pred = gcond["mu_cond"](x_trans_test)
            y_pred = qt_y.inverse_transform(y_trans_pred.reshape(-1, 1))
            return y_pred.reshape(-1)

        trained_models["MG-TFC"] = multi_gauss_trans

    if "MV-LLR" in suite:
        # linear regression with logs.
        # requires shape (n_samples, n_features)
        from sklearn.linear_model import LinearRegression

        reg1 = LinearRegression().fit(np.log(x), np.log(y))

        def linreg(x_test):
            return np.exp(reg1.predict(np.log(x_test)))

        trained_models["MV-LLR"] = linreg

    if "MV-LR" in suite:
        # linear regression (no logs)

        reg2 = LinearRegression().fit(x, y)

        def linreg_no_logs(x_test):
            return reg2.predict(x_test)

        trained_models["MV-LR"] = linreg_no_logs

    if "MV-TLR" in suite:
        # linear regression (no logs)

        reg3 = LinearRegression().fit(x_trans, y_trans)

        def linreg_trans(x_test):
            x_trans_test = qt_x.transform(x_test)
            y_trans_pred = reg3.predict(x_trans_test)
            y_pred = qt_y.inverse_transform(y_trans_pred.reshape(-1, 1))
            return y_pred.reshape(-1)

        trained_models["MV-TLR"] = linreg_trans

    if "CAM" in suite:
        assert NotImplementedError()
        assert "mass_bins" in extra_args and "am_train" in extra_args
        from scipy.interpolate import interp1d

        mass_bins, am_train = extra_args["mass_bins"], extra_args["am_train"]

        a2_train = get_a2_from_am(am_train, mass_bins)

        Y_sort, a2_sort = -np.sort(-y), np.sort(a2_train)
        marks = np.arange(len(Y_sort)) / len(Y_sort)
        marks += (marks[1] - marks[0]) / 2
        a2_to_mark = interp1d(a2_sort, marks, fill_value=(0, 1), bounds_error=False)
        mark_to_Y = interp1d(marks, Y_sort, fill_value=(Y_sort[0], Y_sort[-1]), bounds_error=False)

        def cam(lam_test):
            _a2_test = get_a2_from_am(np.exp(lam_test), mass_bins)
            return mark_to_Y(a2_to_mark(_a2_test))

        trained_models["CAM"] = cam

    if "MG-A2" in suite:
        assert NotImplementedError()

        # multi-normal but just using a_{1/2}
        indx = np.where((0.498 < mass_bins) & (mass_bins < 0.51))[0].item()
        _, _, _, _, mu_cond_a2, _ = gaussian_conditional(
            np.log(Y_train), lam_train[:, indx].reshape(-1, 1)
        )

        def a2_gauss(lam_test):
            return np.exp(mu_cond_a2(lam_test[:, indx].reshape(-1, 1)))

        trained_models["MG-A2"] = a2_gauss

    return trained_models
