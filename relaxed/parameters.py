import numpy as np
from numpy.linalg import norm

default_params = {
    "id",
    "pid",
    "mvir",
    "vvir",
    "rvir",
    "x",
    "y",
    "z",
    "cvir",
    "x0",
    "v0",
    "spin",
    "q",
    "t/|u|",
    "phi_l",
    "gamma_tdyn",
    "tdyn",
    "vmax/vvir",
    "gamma_tdyn",
    "tdyn",
    "scale_of_last_mm",
    "b_to_a",
    "c_to_a",
    "spin_bullock",
}

latex_params = {
    "mvir": {"units": "h^{-1} \\, M_{\\odot}", "form": "M_{\\rm vir}"},
    "tdyn": {"units": "Gyr", "form": "T_{\\rm dyn}"},
    "gamma_tdyn": {
        "units": "h^{-1}\\, yr^{-1} \\, M_{\\odot}",
        "form": "\\gamma_{\\tau_{\\rm dyn}}",
    },
    "t/|u|": {"units": "", "form": "T/|U|"},
    "spin": {"units": "", "form": "\\lambda"},
    "spin_bullock": {"units": "", "form": "\\lambda"},
    "xoff": {"units": "", "form": "x_{\\rm off}"},
    "phi_l": {"units": "", "form": "\\Phi_{l}"},
    "q": {"units": "", "form": "q"},
    "vmax/vvir": {"units": "", "form": "v_{\\rm max} / v_{\\rm vir}"},
    "b/a": {"units": "", "form": "b/a"},
    "c/a": {"units": "", "form": "c/a"},
}


def derive_vvir(mcat, b):
    # units = km/s
    # prevent overflow by combining MKS factors into one constant.
    # C = G_mks * mvir_factor_mks / rvir_factor_mks
    C = 6.674e-11 * 1.988435e30 / 3.086e19
    rvir, mvir = mcat.block(b, ["rvir", "mvir"])
    vvir_mks = np.sqrt(C * mvir / rvir)
    vvir = vvir_mks / 1e3
    return vvir


def derive(pname: str, mcat, b):
    """Derive additional useful halo properties that are not in .minh catalog."""
    available = {"phi_l", "x0", "v0", "tdyn", "eta", "q", "vvir", "cvir", "vmax/vvir"}
    assert pname in available, f"{pname} is not available."

    if pname == "phi_l":
        ax, ay, az, jx, jy, jz = mcat.block(b, ["ax", "ay", "az", "jx", "jy", "jz"])
        num = ax*jx + ay*jy + az*jz
        denom = np.sqrt(ax**2 + ay**2 + az**2) * np.sqrt(jx**2 + jy**2 + jz**2)
        return np.arccos(num / denom)

    if pname == "x0":
        xoff, rvir = mcat.block(b, ["xoff", "rvir"])
        return xoff / rvir

    if pname == "v0":
        vvir = derive_vvir(mcat, b)
        [voff] = mcat.block(b, ["voff"])  # km /s
        return voff / vvir

    if pname == "tdyn":
        [rvir] = mcat.block(b, ["rvir"])
        vvir = derive_vvir(mcat, b)
        rvir_mks = rvir * 3.086e19  # rvir in kpc/h not Mpc/h
        vvir_mks = vvir * 1e3
        tdyn_mks = 2 * rvir_mks / vvir_mks
        return tdyn_mks / (365 * 24 * 3600) / 10**9

    if pname == "q":
        b_to_a, c_to_a = mcat.block(b, ["b_to_a", "c_to_a"])
        return (1 / 2) * (b_to_a + c_to_a)

    if pname == "vvir":
        return derive_vvir(mcat, b)

    if pname == "cvir":
        rvir, rs = mcat.block(b, ["rvir", "rs"])
        return rvir / rs

    if pname == "vmax/vvir":
        vvir = derive_vvir(mcat, b)
        [vmax] = mcat.block(b, ["vmax"])
        return vmax / vvir

    else:
        raise NotImplementedError()
