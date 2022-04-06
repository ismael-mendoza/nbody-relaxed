import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from relaxed.correlations import vol_jacknife_err


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


def compute_metrics(pred_func, _x_test, _y_test, box_keep=None):
    # pred_func should return ONLY 1 parameter. (e.g. lambda wrapped around indexing)
    if box_keep is None:
        box_keep = np.ones(_x_test.shape[0]).astype(bool)

    x_test = _x_test[box_keep]
    y_test = _y_test[box_keep].reshape(-1)

    y_pred = pred_func(x_test).reshape(-1)
    x = (y_pred - y_test) / np.std(y_test)  # normalize

    return {
        "mu": np.mean(x),
        "sigma_ratio": np.std(y_pred) / np.std(y_test),
        "spear": spearmanr(y_pred, y_test)[0],
        "rscatter": np.std(x) / np.sqrt(2),
        "mad": np.mean(np.abs(x)),
    }


def metrics_plot(
    metrics_data: dict,
    test_data: dict,
    trained_models: dict,
    cat_test,
    params=("cvir",),
    figsize=(12, 12),
    nrows=1,
    ncols=1,
    ticksize=16,
    bbox_to_anchor=(0.0, 1.0, 0.3, 0.3),
    y_label_size=28,
    ms=10,
):
    # NOTE: params and models MUST have same order
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    model_metrics = {}

    for jj, param in enumerate(params):
        for mdl_name, (x_test, label, color, _) in test_data.items():
            model = trained_models[mdl_name]
            pred_func = lambda x: model.predict(x)[:, jj]  # noqa: E731
            y_test = cat_test[param].value.reshape(-1)
            val_metrics = compute_metrics(pred_func, x_test, y_test)
            errs = vol_jacknife_err(
                cat_test, compute_metrics, pred_func, x_test, y_test, mode="dict"
            )
            model_metrics[(param, mdl_name)] = {k: (val_metrics[k], errs[k]) for k in metrics_data}

    params_latex = [latex_params[par] for par in params]
    for ii, met in enumerate(metrics_data):
        ax = axes[ii]
        ax.set_xlim(-0.25, len(params))
        if "yrange" in metrics_data[met]:
            if metrics_data[met]["yrange"] is not None:
                ax.set_ylim(metrics_data[met]["yrange"])
        ax.set_xticks(np.array(list(range(len(params)))))
        ax.set_xticklabels(params_latex)
        ax.tick_params(axis="x", labelsize=ticksize)
        x_bias = 0.0
        for mdl_name, (x_test, label, color, marker) in test_data.items():
            for jj, param in enumerate(params):
                label = label if (jj == 0 and ii == 0) else None
                val, err = model_metrics[(param, mdl_name)][met]
                ax.errorbar(
                    jj + x_bias,
                    val,
                    yerr=err,
                    label=label,
                    fmt=marker,
                    color=color,
                    capsize=2.5,
                    ms=ms,
                    capthick=2.0,
                )
            x_bias += 0.1

        ax.set_ylabel(latex_metrics[met], size=y_label_size)
        if metrics_data[met].get("hline", None) is not None:
            ax.axhline(metrics_data[met]["hline"], ls="--", color="k")

        if ii == 0:
            ax.legend(loc="lower left", bbox_to_anchor=bbox_to_anchor)
    plt.tight_layout()
    return fig
