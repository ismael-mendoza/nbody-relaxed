import numpy as np
from astropy.table import Table

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
    available = {"phi_l", "x0", "v0", "tdyn", "eta", "q", "vvir", "cvir"}
    assert pname in available

    if pname == "phi_l":
        pass

    if pname == "x0":
        pass

    if pname == "v0":
        pass

    if pname == "tdyn":
        rvir = mcat.block(b, "rvir")
        vvir = derive_vvir(mcat, b)
        rvir_mks = rvir * 3.086e19  # rvir in kpc/h not Mpc/h
        vvir_mks = vvir * 1e3
        tdyn_mks = 2 * rvir_mks / vvir_mks
        return tdyn_mks / (365 * 24 * 3600) / 10**9

    if pname == "eta":
        pass

    if pname == "q":
        pass

    if pname == "vvir":
        pass

    if pname == "cvir":
        rvir, rs = mcat.block(b, ["rvir", "rs"])
        return rvir / rs

    else:
        raise NotImplementedError()


class BToA(HaloParam):
    units = ""

    @property
    def name(self):
        return "b_to_a"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "b/a",
        }


class CToA(HaloParam):
    units = ""

    @property
    def name(self):
        return "c_to_a"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "c/a",
        }


class Cvir_Klypin(HaloParam):
    units = ""

    @property
    def name(self):
        return "cvir_klypin"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "c_{\\rm vir}",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: cat["rvir"] / cat["rs_klypin"],
            "requires": ("rvir", "rs_klypin"),
        }


class Eta(HaloParam):
    units = ""

    @property
    def name(self):
        return "eta"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "\\eta",
        }

    @property
    def derive(self):
        return {"func": lambda cat: 2 * cat["t/|u|"], "requires": ("t/|u|",)}


class Q(HaloParam):
    units = ""

    @property
    def name(self):
        return "q"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "q",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: (1 / 2) * (cat["b_to_a"] + cat["c_to_a"]),
            "requires": ("b_to_a", "c_to_a"),
        }


class Phi_L(HaloParam):
    units = ""

    @staticmethod
    def get_phi_l(cat):
        """
        * JX/JY/JZ: Halo angular momenta ((Msun/h) * (Mpc/h) * km/s (physical))
        * A[x],A[y],A[z]: Largest shape ellipsoid axis (kpc/h comoving)
        :return: Value of phi_l for each row of the catalog cat.
        :rtype: astropy.Column
        """
        numerator = cat["ax"] * cat["jx"] + cat["ay"] * cat["jy"] + cat["az"] * cat["jz"]
        denominator = np.sqrt(cat["ax"] ** 2 + cat["ay"] ** 2 + cat["az"] ** 2)
        denominator *= np.sqrt(cat["jx"] ** 2 + cat["jy"] ** 2 + cat["jz"] ** 2)
        return np.arccos(numerator / denominator)

    @property
    def name(self):
        return "phi_l"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "\\Phi_{l}",
        }

    @property
    def derive(self):
        return {
            "func": self.get_phi_l,
            "requires": ("ax", "ay", "az", "jx", "jy", "jz"),
        }


class X0(HaloParam):
    units = ""

    @property
    def name(self):
        return "x0"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "x_{\\rm off}",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: cat["xoff"] / cat["rvir"],
            "requires": ("xoff", "rvir"),
        }


class V0(HaloParam):
    units = ""

    @property
    def name(self):
        return "v0"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "v_{\\rm off}",
        }

    @property
    def derive(self):
        return {
            "func": lambda cat: cat["voff"] / Vvir.calc_vvir(cat),
            "requires": ("mvir", "rvir", "voff"),
        }


class Fsub(HaloParam):
    units = ""

    @property
    def name(self):
        return "f_sub"

    @property
    def latex(self):
        return {
            "units": "",
            "form": "f_{\\rm sub}",
        }

    def get_values_minh(self, mcat):
        raise NotImplementedError("Cannot obtain f_sub from minh")

    def get_values_minh_block(self, mcat, b=None):
        raise NotImplementedError("Cannot obtain f_sub from minh")
