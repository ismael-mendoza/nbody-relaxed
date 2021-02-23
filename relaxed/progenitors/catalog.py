import numpy as np
from scipy.optimize import curve_fit


def lma_fit(z, alpha):
    return -alpha * z


def log_m_a_fit_ab(z, alpha, beta):
    return beta * np.log(1 + z) - alpha * z


def get_alpha(zs, lma):
    # use the fit of the form:
    # log m(z) = - \alpha * z
    # get best exponential fit to the line of main progenitors.

    opt_params, _ = curve_fit(lma_fit, zs, lma, p0=(1,))
    return opt_params  # = alpha


def get_alpha_beta(zs, log_m_a):
    # use the fit of the ofrm M(z) = M(0) * (1 + z)^{\beta} * exp(- \gamma * z)
    # get best exponential fit to the line of main progenitors.
    opt_params, _ = curve_fit(log_m_a_fit_ab, zs, log_m_a, p0=(1, 1))
    return opt_params  # = alpha, beta
