import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-colorblind")


mpl.rcParams.update(
    {
        # figure
        "figure.figsize": (10, 10),
        # axes
        "axes.labelsize": 24,
        "axes.titlesize": 28,
        # ticks
        "xtick.major.size": 10,
        "xtick.minor.size": 5,
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "xtick.labelsize": 22,
        "ytick.major.size": 10,
        "ytick.minor.size": 5,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "ytick.labelsize": 22,
        # legend
        "legend.fontsize": 22,
    }
)
plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
plt.rc("text", usetex=True)


default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cb_colors = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

latex_params = {
    "cvir": r"$c_{\rm vir}$",
    "t/|u|": r"$t/\vert u \vert$",
    "x0": r"$x_{\rm off}$",
    "spin": r"$\lambda$",
    "q": r"$q$",
    "spin_bullock": r"$\lambda_{\rm bullock}$",
    "b_to_a": r"$b/a$",
    "c_to_a": r"$c/a$",
    "cvir_klypin": r"$c_{\rm vir, klypin}$",
    "a2": r"$a_{1/2}$",
    "a4": r"$a_{3/4}$",
    "alpha": r"$\alpha$",
    "mdyn": r"$m(a_{\rm dyn})$",
    "tau_c": r"$\tau_{c}$",
    "alpha_early": r"$\alpha_{\rm early}$",
    "alpha_late": r"$\alpha_{\rm late}$",
}

latex_metrics = {
    "mu": r"$\mu \left( y_{\rm pred} - y_{\rm true} \right)$",
    "med": r"$\mu_{x}'$",
    "sigma_ratio": r"$\sigma_{\rm pred} / \sigma_{\rm true}$",
    "spear": r"$\rho_{\rm spearman} \left(y_{\rm true}, y_{\rm pred} \right)$",
    "rscatter": r"$\frac{\sigma(y_{\rm pred} - y_{\rm true})}{ \sigma(y_{\rm true}) \sqrt{2}}$",
    "mad": r"\rm MAD",
}


def draw_histogram(
    ax,
    values,
    n_bins=30,
    bins=None,
    vline="median",
    color="r",
    **hist_kwargs,
):
    ax.hist(
        values,
        bins=bins if bins is not None else n_bins,
        histtype="step",
        color=color,
        **hist_kwargs,
    )

    # add a vertical line.
    if vline == "median":
        ax.axvline(np.median(values), ls="--", color=color)

    elif isinstance(vline, float) or isinstance(vline, int):
        ax.axvline(vline, ls="--", color=color)

    elif vline is None:
        pass

    else:
        raise NotImplementedError(
            f"vline type {type(vline)} is not compatible with current implementation."
        )
