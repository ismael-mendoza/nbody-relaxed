#!/usr/bin/env python3
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from relaxed import plot_defaults as plotdf
from relaxed.correlations import get_am_corrs
from relaxed.correlations import get_ma_corrs
from relaxed.cosmo import get_fractional_tdyn
from relaxed.mah import get_mah
from relaxed.models import opcam_dict
from relaxed.models import prepare_datasets
from relaxed.models import training_suite

plt.ioff()


root = Path(__file__).absolute().parent.parent
figsdir = root.joinpath("results", "figs")
figsdir.mkdir(exist_ok=True, parents=False)


def make_correlation_mah_plots():
    params = ["cvir", "cvir_klypin", "x0", "t/|u|", "spin", "spin_bullock", "q", "b_to_a", "c_to_a"]
    lss = np.array(["-", "--"])

    # load data
    mahdir = root.joinpath("data", "processed", "bolshoi_m12")
    mah_data = get_mah(mahdir, cutoff_missing=0.05, cutoff_particle=0.05)
    cat = mah_data["cat"]
    scales = mah_data["scales"]
    am = mah_data["am"]
    ma = mah_data["ma"]
    mass_bins = mah_data["mass_bins"]

    # remove last index to avoid correlation warning.
    ma = ma[:, :-1]
    scales = scales[:-1]
    am = am[:, :-1]
    mass_bins = mass_bins[:-1]

    # (1) Correlation with m(a)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    tdyn = np.mean(cat["tdyn"]) / 10**9  # Gyr which astropy also returns by default
    max_dict = {}

    for j, param in enumerate(params):
        latex_param = plotdf.latex_params[param]
        color = plotdf.cb_colors[j]
        corrs = get_ma_corrs(cat, param, ma)
        pos = corrs > 0
        neg = ~pos
        _corrs = abs(corrs)
        max_indx = np.nanargmax(_corrs)
        max_dict[param] = corrs[max_indx], scales[max_indx]

        # plot positive corr and negative corr with different markers.
        if sum(pos) > 0:
            label = f"${latex_param}$" if sum(pos) > sum(neg) else None
            ax.plot(scales[pos], _corrs[pos], color=color, ls=lss[0], label=label, markersize=7)

        if sum(neg) > 0:
            label = f"${latex_param}$" if sum(pos) < sum(neg) else None
            ax.plot(scales[neg], _corrs[neg], color=color, ls=lss[1], label=label, markersize=7)

    # draw a vertical line at max scales
    text = ""
    for j, param in enumerate(params):
        corr, scale = max_dict[param]
        color = plotdf.cb_colors[j]
        ax.axvline(scale, linestyle="--", color=color)
        text += f"{param}: Max corr is {corr:.3f} at scale {scale:.3f}\n"

    with open(figsdir.joinpath("corrs_ma.txt"), "w") as fp:
        print(text.strip(), file=fp)

    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_ylabel("$\\rho(\\cdot, m(a))$", size=22)
    ax.set_xlabel("$a$", size=22)
    ax.tick_params(axis="both", which="major", labelsize=16)

    # add additional x-axis with tydn fractional scale
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())

    xticks = ax.get_xticks()[1:]
    fractional_tdyn = get_fractional_tdyn(xticks, tdyn, sim_name="Bolshoi")
    fractional_tdyn = [f"${x/10**9:.2g}$" for x in fractional_tdyn]
    ax2.set_xticklabels([np.nan] + fractional_tdyn, size=16)
    ax2.set_xlabel("$\\tau_{\\rm dyn} \\, {\\rm [Gyrs]}$", size=22, labelpad=10)

    ax.legend(loc="best", prop={"size": 16})

    ax.set_xlim(0.15, 1)
    ax2.set_xlim(0.15, 1)

    fig.savefig(figsdir.joinpath("corrs_ma.png"))

    ###############################################################################################
    # (2) Correlations with a(m)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    max_dict = {}

    for j, param in enumerate(params):
        latex_param = plotdf.latex_params[param]
        color = plotdf.cb_colors[j]
        corrs = get_am_corrs(cat, param, am)
        pos = corrs >= 0
        neg = ~pos
        _corrs = abs(corrs)
        max_dict[param] = corrs[np.nanargmax(_corrs)], mass_bins[np.nanargmax(_corrs)]

        # plot positive corr and negative corr with different markers.
        if sum(pos) > 0:
            label = f"${latex_param}$" if sum(pos) > sum(neg) else None
            ax.plot(mass_bins[pos], _corrs[pos], color=color, ls=lss[0], label=label)

        if sum(neg) > 0:
            label = f"${latex_param}$" if sum(pos) < sum(neg) else None
            ax.plot(mass_bins[neg], _corrs[neg], color=color, ls=lss[1], label=label)

    # draw a vertical line at max scales, output table.
    text = ""
    for j, param in enumerate(params):
        color = plotdf.cb_colors[j]
        corr, mbin = max_dict[param]
        ax.axvline(mbin, linestyle="--", color=color)
        text += f"{param}: Max corr is {corr:.3f} at mass bin {mbin:.3f}\n"
    with open(figsdir.joinpath("corrs_am.txt"), "w") as fp:
        print(text.strip(), file=fp)

    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.01, 1.0)
    ax.set_ylabel("$\\rho(\\cdot, a(m))$", size=22)
    ax.set_xlabel("$m$", size=22)
    ax.tick_params(axis="both", which="major", labelsize=16, size=10)
    ax.tick_params(axis="x", which="minor", size=8)
    ax.legend(loc="best", prop={"size": 16})

    plt.tight_layout()
    fig.savefig(figsdir.joinpath("corrs_am.png"))


def make_triangle(model_name, params, trained_models, datasets, sample_fn, figfile):
    x_test = datasets["all"]["test"][0]
    model = trained_models[model_name]
    y_true = datasets["all"]["test"][1]
    y_samples = sample_fn(model, x_test)

    # ellipticty parameters look better not logged
    y1 = np.log(y_true)
    y1[:, -3:] = np.exp(y1[:, -3:])

    y2 = np.log(y_samples)
    y2[:, -3:] = np.exp(y2[:, -3:])

    labels = [plotdf.latex_params[param] for param in params]
    fig = corner.corner(y1, labels=labels, max_n_ticks=3, color="C1", labelpad=0.05)
    fig = corner.corner(y2, labels=labels, max_n_ticks=3, fig=fig, color="C2", labelpad=0.05)

    fig.savefig(figfile)


def make_triangle_plots():
    params = ("cvir", "cvir_klypin", "t/|u|", "x0", "spin", "spin_bullock", "q", "b_to_a", "c_to_a")
    mahdir = root.joinpath("data", "processed", "bolshoi_m12")
    mah_data = get_mah(mahdir, cutoff_missing=0.05, cutoff_particle=0.05)

    cat = mah_data["cat"]
    am = mah_data["am"]
    mass_bins = mah_data["mass_bins"]

    # prepare catalog with all a_m
    am_names = [f"am_{ii}" for ii in range(len(mass_bins))]
    for ii in range(len(mass_bins)):
        cat.add_column(am[:, ii], name=am_names[ii])

    # dataset preparation
    info = {"all": {"x": am_names, "y": params}}
    datasets, _, _ = prepare_datasets(cat, info)

    # models to use

    data = {
        "multicam": {
            "xy": datasets["all"]["train"],
            "n_features": 100,
            "n_targets": len(params),
            "model": "gaussian",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "optcam": {
            "xy": datasets["all"]["train"],
            "n_features": 100,
            "n_targets": len(params),
            "model": "mixed_cam",
            "kwargs": {
                "mass_bins": mass_bins,
                "opt_mbins": [opcam_dict[param]["mbin"] for param in params],
                "cam_orders": [opcam_dict[param]["order"] for param in params],
            },
        },
    }
    joint_models = training_suite(data)

    # (1) MultiGaussian sample triangle
    figfile = figsdir.joinpath("triangle_multigaussian.png")
    sample_fn = lambda model, x: model.sample(x, 1).reshape(-1, model.n_targets)
    make_triangle("multicam", params, joint_models, datasets, sample_fn, figfile)

    # (2) LR pred triangle
    figfile = figsdir.joinpath("triangle_lr.png")
    sample_fn = lambda model, x: model.predict(x).reshape(-1, model.n_targets)
    make_triangle("multicam", params, joint_models, datasets, sample_fn, figfile)

    # (3) OptCAM pred triangle
    figfile = figsdir.joinpath("triangle_cam.png")
    sample_fn = lambda model, x: model.predict(x).reshape(-1, model.n_targets)
    make_triangle("optcam", params, joint_models, datasets, sample_fn, figfile)


def make_figures():
    make_correlation_mah_plots()
    make_triangle_plots()


if __name__ == "__main__":
    make_figures()
