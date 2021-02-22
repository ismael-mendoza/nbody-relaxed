import numpy as np
from scipy.optimize import curve_fit


def lma_fit(z, alpha):
    return -alpha * z


def get_alpha(zs, lma):
    # use the fit of the form:
    # log m(z) = - \alpha * z
    # get best exponential fit to the line of main progenitors.

    opt_params, _ = curve_fit(lma_fit, zs, lma, p0=(1,))
    return opt_params  # = alpha


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
    import scipy

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
            scipy.interpolate.interp1d(
                np.log(_Mas), np.log(_scales), bounds_error=False, fill_value=np.nan
            )
        )

    # 6.
    am = np.array([np.exp(f(mass_bins)) for f in fs])

    return am, np.exp(mass_bins)


def log_m_a_fit_ab(z, alpha, beta):
    return beta * np.log(1 + z) - alpha * z


def get_alpha_beta(zs, log_m_a):
    # use the fit of the ofrm M(z) = M(0) * (1 + z)^{\beta} * exp(- \gamma * z)
    # get best exponential fit to the line of main progenitors.
    from scipy.optimize import curve_fit

    opt_params, _ = curve_fit(log_m_a_fit_ab, zs, log_m_a, p0=(1, 1))

    return opt_params  # = alpha, beta
