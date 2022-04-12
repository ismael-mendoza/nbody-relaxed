#!/usr/bin/env python3
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.stats import spearmanr

from relaxed import plotting as rxplots
from relaxed.correlations import get_am_corrs
from relaxed.correlations import get_ma_corrs
from relaxed.cosmo import get_a_from_t
from relaxed.cosmo import get_fractional_tdyn
from relaxed.cosmo import get_t_from_a
from relaxed.fits import alpha_analysis
from relaxed.fits import get_early_late
from relaxed.gradients import get_savgol_grads
from relaxed.mah import get_an_from_am
from relaxed.mah import get_mah
from relaxed.models import opcam_dict
from relaxed.models import prepare_datasets
from relaxed.models import training_suite
from relaxed.plotting import CB_COLORS
from relaxed.plotting import MARKS
from relaxed.plotting import set_rc

plt.ioff()


root = Path(__file__).absolute().parent.parent
figsdir = root.joinpath("results", "figs")
figsdir.mkdir(exist_ok=True, parents=False)


def make_correlation_mah_plots():
    set_rc(figsize=(7, 7), fontsize=24, lgsize=14)
    params = ["cvir", "cvir_klypin", "x0", "t/|u|", "spin_bullock", "b_to_a", "c_to_a"]
    lss = np.array(["-", ":"])

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
    fig, ax = plt.subplots(1, 1)
    tdyn = np.mean(cat["tdyn"]) / 10**9  # Gyr which astropy also returns by default
    max_dict = {}

    for j, param in enumerate(params):
        latex_param = rxplots.latex_params[param]
        color = CB_COLORS[j]
        corrs = get_ma_corrs(cat, param, ma)
        pos = corrs > 0
        neg = ~pos
        _corrs = abs(corrs)
        max_indx = np.nanargmax(_corrs)
        max_dict[param] = corrs[max_indx], scales[max_indx]

        # plot positive corr and negative corr with different markers.
        if sum(pos) > 0:
            label = f"${latex_param}$" if sum(pos) > sum(neg) else None
            ax.plot(scales[pos], _corrs[pos], color=color, ls=lss[0], label=label)

        if sum(neg) > 0:
            label = f"${latex_param}$" if sum(pos) < sum(neg) else None
            ax.plot(scales[neg], _corrs[neg], color=color, ls=lss[1], label=label)

    # draw a vertical line at max scales
    text = ""
    for j, param in enumerate(params):
        corr, scale = max_dict[param]
        color = CB_COLORS[j]
        ax.axvline(scale, linestyle="--", color=color)
        text += f"{param}: Max corr is {corr:.3f} at scale {scale:.3f}\n"

    with open(figsdir.joinpath("corrs_ma.txt"), "w") as fp:
        print(text.strip(), file=fp)

    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 1.0)
    ax.set_ylabel("$\\rho(\\cdot, m(a))$")
    ax.set_xlabel("$a$")

    # add additional x-axis with tydn fractional scale
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())

    xticks = ax.get_xticks()[1:]
    fractional_tdyn = get_fractional_tdyn(xticks, tdyn, sim_name="Bolshoi")
    fractional_tdyn = [f"${x/10**9:.2g}$" for x in fractional_tdyn]
    ax2.set_xticklabels([np.nan] + fractional_tdyn)
    ax2.set_xlabel("$\\tau_{\\rm dyn} \\, {\\rm [Gyrs]}$", labelpad=10)

    ax.legend(loc="upper right")

    ax.set_xlim(0.15, 1)
    ax2.set_xlim(0.15, 1)

    fig.savefig(figsdir.joinpath("corrs_ma.png"))

    ###############################################################################################
    # (2) Correlations with a(m)
    fig, ax = plt.subplots(1, 1)
    max_dict = {}

    for j, param in enumerate(params):
        latex_param = rxplots.latex_params[param]
        color = CB_COLORS[j]
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
        color = CB_COLORS[j]
        corr, mbin = max_dict[param]
        ax.axvline(mbin, linestyle="--", color=color)
        text += f"{param}: Max corr is {corr:.3f} at mass bin {mbin:.3f}\n"
    with open(figsdir.joinpath("corrs_am.txt"), "w") as fp:
        print(text.strip(), file=fp)

    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.01, 1.0)
    ax.set_ylabel("$\\rho(\\cdot, a(m))$")
    ax.set_xlabel("$m$")
    ax.tick_params(axis="both", which="major")
    ax.tick_params(axis="x", which="minor")
    ax.legend(loc="upper right")
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

    labels = [rxplots.latex_params[param] for param in params]
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
    sample_fn = lambda model, x: model.sample(x, 1).reshape(-1, model.n_targets)  # noqa: E731
    make_triangle("multicam", params, joint_models, datasets, sample_fn, figfile)

    # (2) LR pred triangle
    figfile = figsdir.joinpath("triangle_lr.png")
    sample_fn = lambda model, x: model.predict(x).reshape(-1, model.n_targets)  # noqa: E731
    make_triangle("multicam", params, joint_models, datasets, sample_fn, figfile)

    # (3) OptCAM pred triangle
    figfile = figsdir.joinpath("triangle_cam.png")
    sample_fn = lambda model, x: model.predict(x).reshape(-1, model.n_targets)  # noqa: E731
    make_triangle("optcam", params, joint_models, datasets, sample_fn, figfile)


def make_mah_pred_plots():
    set_rc(fontsize=28, lgsize=18, figsize=(9, 6))
    mahdir = root.joinpath("data", "processed", "bolshoi_m12")
    mah_data = get_mah(mahdir, cutoff_missing=0.05, cutoff_particle=0.05)

    cat = mah_data["cat"]
    ma = mah_data["ma"]
    am = mah_data["am"]
    mass_bins = mah_data["mass_bins"][:-1]  # remove last bin to avoid spearman error.
    scales = mah_data["scales"][:-1]  # same for scales.
    n_mbins = len(mass_bins)
    n_scales = len(scales)

    # prepare catalog with all m_a
    ma_names = [f"ma_{ii}" for ii in range(len(scales))]
    for ii in range(len(scales)):
        cat.add_column(ma[:, ii], name=ma_names[ii])

    # prepare catalog with all a(m)
    am_names = [f"am_{ii}" for ii in range(len(mass_bins))]
    for ii in range(len(mass_bins)):
        cat.add_column(am[:, ii], name=am_names[ii])

    # prepare datasets
    info = {
        "cvir_only": {"x": ("cvir",), "y": am_names + ma_names},
        "x0_only": {"x": ("x0",), "y": am_names + ma_names},
        "tu_only": {"x": ("t/|u|",), "y": am_names + ma_names},
        "all": {
            "x": ("cvir", "cvir_klypin", "t/|u|", "x0", "spin_bullock", "c_to_a", "b_to_a"),
            "y": am_names + ma_names,
        },
    }
    datasets, _, _ = prepare_datasets(cat, info)

    # train models
    data = {
        "linear_cvir": {
            "xy": datasets["cvir_only"]["train"],
            "n_features": 1,
            "n_targets": n_mbins + n_scales,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "linear_x0": {
            "xy": datasets["x0_only"]["train"],
            "n_features": 1,
            "n_targets": n_mbins + n_scales,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "linear_tu": {
            "xy": datasets["tu_only"]["train"],
            "n_features": 1,
            "n_targets": n_mbins + n_scales,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "linear_all": {
            "xy": datasets["all"]["train"],
            "n_features": 7,
            "n_targets": n_mbins + n_scales,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
    }
    models = training_suite(data)

    corrs_am = {}
    corrs_ma = {}
    dataset_names = ["cvir_only", "x0_only", "tu_only", "all"]
    mdl_names = ["linear_cvir", "linear_x0", "linear_tu", "linear_all"]
    for dataset_names, mdl_name in zip(dataset_names, mdl_names):
        model = models[mdl_name]
        x_test, y_test = datasets[dataset_names]["test"]
        y_pred = model.predict(x_test)
        corrs_am[mdl_name] = np.array(
            [spearmanr(y_pred[:, jj], y_test[:, jj]).correlation for jj in range(n_mbins)]
        )
        corrs_ma[mdl_name] = np.array(
            [
                spearmanr(y_pred[:, jj + n_mbins], y_test[:, jj + n_mbins]).correlation
                for jj in range(n_scales)
            ]
        )

    # (1) Correlation a(m) vs a_pred(m) figure
    fig, ax = plt.subplots(1, 1)
    nice_names = [
        r"\rm MultiCAM $c_{\rm vir}$ only",
        r"\rm MultiCAM $x_{\rm off}$ only",
        r"\rm MultiCAM $2T / \vert U \vert$ only",
        r"\rm MultiCAM all parameters",
    ]
    for nice_name, mdl_name in zip(nice_names, mdl_names):
        ax.plot(mass_bins, corrs_am[mdl_name], label=nice_name)
    ax.set_xlabel("$m$")
    ax.set_ylabel(r"$\rho(a_{m}, a_{m, \rm{pred}})$")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(f"${x:.1f}$" for x in ax.get_yticks())
    ax.legend()
    figfile = figsdir.joinpath("corr_pred_mah_am.png")
    fig.savefig(figfile)

    # (2) Correlation m(a) vs m_pred(a) figure
    fig, ax = plt.subplots(1, 1)
    for nice_name, mdl_name in zip(nice_names, mdl_names):
        ax.plot(scales, corrs_ma[mdl_name], label=nice_name)
    ax.set_xlabel("$a$")
    ax.set_ylabel(r"$\rho(m_{a}, m_{a, \rm{pred}})$")
    ax.legend()
    figfile = figsdir.joinpath("corr_pred_mah_ma.png")
    fig.savefig(figfile)


def make_inv_pred_plots():
    set_rc()
    mahdir = root.joinpath("data", "processed", "bolshoi_m12")
    mah_data = get_mah(mahdir, cutoff_missing=0.05, cutoff_particle=0.05)

    cat = mah_data["cat"]
    ma = mah_data["ma"]
    am = mah_data["am"]
    scales = mah_data["scales"]
    mass_bins = mah_data["mass_bins"]

    # extract m(a_{tdyn}) = \dot{M}_dynamical
    t0 = get_t_from_a(1)
    tdyn = np.mean(cat["tdyn"])
    t = (t0 - tdyn) * u.Gyr
    a_dyn = get_a_from_t(t)
    indx_dyn = np.where(scales > a_dyn)[0][0]
    mdyn = ma[:, indx_dyn].reshape(-1, 1)

    # extract alpha fits
    alpha_file = root.joinpath("data", "processed", "alpha_fits.npy")
    alphas, _, _ = alpha_analysis(ma, scales, mass_bins, alpha_file=alpha_file)

    # extract a_{1/2} and a_{3/4}
    a2 = get_an_from_am(am, mass_bins, mbin=0.5)
    a4 = get_an_from_am(am, mass_bins, mbin=0.75)

    pars = np.load(root.joinpath("data", "processed", "pbest_diffmah.npy"))
    logtc, ue, ul = pars[:, 0], pars[:, 1], pars[:, 2]
    early, late = get_early_late(ue, ul)

    # add everything extracted to catalog
    cat.add_column(10**logtc, name="tau_c")
    cat.add_column(early, name="alpha_early")
    cat.add_column(late, name="alpha_late")
    cat.add_column(alphas, name="alpha")
    cat.add_column(a2, name="a2")
    cat.add_column(a4, name="a4")
    cat.add_column(mdyn, name="mdyn")

    params = ("a2", "a4", "alpha", "mdyn", "tau_c", "alpha_late", "alpha_early")
    info = {
        "cvir_only": {
            "x": ("cvir",),
            "y": params,
        },
        "x0_only": {
            "x": ("x0",),
            "y": params,
        },
        "tu_only": {
            "x": ("t/|u|",),
            "y": params,
        },
        "all": {
            "x": ("cvir", "cvir_klypin", "t/|u|", "x0", "spin_bullock", "c_to_a", "b_to_a"),
            "y": params,
        },
    }
    datasets, _, cat_test = prepare_datasets(cat, info)

    data = {
        "linear_cvir": {
            "xy": datasets["cvir_only"]["train"],
            "n_features": 1,
            "n_targets": 7,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "linear_x0": {
            "xy": datasets["x0_only"]["train"],
            "n_features": 1,
            "n_targets": 7,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "linear_tu": {
            "xy": datasets["tu_only"]["train"],
            "n_features": 1,
            "n_targets": 7,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "linear_all": {
            "xy": datasets["all"]["train"],
            "n_features": 7,
            "n_targets": 7,
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
    }
    models = training_suite(data)

    # make metrics plot
    metrics_data = {
        "spear": {"yrange": (0, 0.85)},
        # "rscatter": {},
        # "sigma_ratio": {"yrange": (0.8, 1.2), "hline": 1.0},
        # "mu": {"yrange": (-0.1, 0.1), "hline": 0.0},
    }

    test_data = {
        "linear_all": (
            datasets["all"]["test"][0],
            r"\rm MultiCAM",
            CB_COLORS[0],
            MARKS[0],
        ),
        "linear_cvir": (
            datasets["cvir_only"]["test"][0],
            r"\rm MultiCAM $c_{\rm vir}$ only",
            CB_COLORS[1],
            MARKS[1],
        ),
        "linear_x0": (
            datasets["x0_only"]["test"][0],
            r"\rm MultiCAM $x_{\rm off}$ only",
            CB_COLORS[2],
            MARKS[2],
        ),
        "linear_tu": (
            datasets["tu_only"]["test"][0],
            r"\rm MultiCAM $T/\vert U \vert$ only",
            CB_COLORS[3],
            MARKS[3],
        ),
    }
    fig = rxplots.metrics_plot(metrics_data, test_data, models, cat_test, params)
    figfile = figsdir.joinpath("inv_pred.png")
    fig.savefig(figfile)


def make_pred_plots():
    set_rc()
    params = ("cvir", "cvir_klypin", "t/|u|", "x0", "b_to_a", "c_to_a", "spin_bullock")
    mahdir = root.joinpath("data", "processed", "bolshoi_m12")
    mah_data = get_mah(mahdir, cutoff_missing=0.05, cutoff_particle=0.05)

    cat = mah_data["cat"]
    ma = mah_data["ma"]
    am = mah_data["am"]
    scales = mah_data["scales"]
    mass_bins = mah_data["mass_bins"]

    # prepare catalog with all m_a
    ma_names = [f"ma_{ii}" for ii in range(len(scales))]
    for ii in range(len(scales)):
        cat.add_column(ma[:, ii], name=ma_names[ii])

    # prepare catalog with all a_m
    am_names = [f"am_{ii}" for ii in range(len(mass_bins))]
    for ii in range(len(mass_bins)):
        cat.add_column(am[:, ii], name=am_names[ii])

    # load alpha fits
    alpha_file = root.joinpath("data", "processed", "alpha_fits.npy")
    alphas, _, _ = alpha_analysis(ma, scales, mass_bins, alpha_file=alpha_file)
    cat.add_column(alphas, name="alpha")

    # load diffmah parameters of best fits.
    pars = np.load(root.joinpath("data", "processed", "pbest_diffmah.npy"))
    logtc, ue, ul = pars[:, 0], pars[:, 1], pars[:, 2]
    early, late = get_early_late(ue, ul)
    cat.add_column(10**logtc, name="tau_c")
    cat.add_column(early, name="alpha_early")
    cat.add_column(late, name="alpha_late")

    # load diffmah fits curves.
    diffmah_curves = np.load(root.joinpath("data", "processed", "diffmah_fits.npy"))
    ma_diffmah_names = [f"ma_diffmah_{ii}" for ii in range(len(scales))]
    for ii in range(len(scales)):
        cat.add_column(diffmah_curves[:, ii], name=ma_diffmah_names[ii])

    # add a_{1/2} also as alternative parametrization
    cat.add_column(get_an_from_am(am, mass_bins, 0.5), name="a2")

    # add savitsky-golay gradients
    ks = [11, 21, 41, 81, 121, 161]
    log_a = np.log(scales)
    delta = abs(log_a[-1] - log_a[0]) / (
        200 - 1
    )  # 200 is default number of interpolation points for uniform spacing. (in get_savgol_grads)
    gamma_k = {k: -get_savgol_grads(scales, ma, k=k) for k in ks}
    delta_k = {k: delta * (k // 2) for k in ks}
    grad_names_k = {k: [f"grad_{k}_{jj}" for jj in range(gamma_k[k].shape[1])] for k in ks}
    all_grad_names = [
        grad_names_k[k][jj] for k in grad_names_k for jj in range(len(grad_names_k[k]))
    ]
    assert delta_k and all_grad_names

    # add gradients to catalog catalog
    for k in ks:
        for jj in range(gamma_k[k].shape[1]):
            name = grad_names_k[k][jj]
            value = gamma_k[k][:, jj]
            cat.add_column(value, name=name)

    info = {
        "ma": {
            "x": ma_names,
            "y": params,
        },
        "am": {
            "x": am_names,
            "y": params,
        },
        "ma_diffmah": {
            "x": ma_diffmah_names,
            "y": params,
        },
        "params_diffmah": {
            "x": ("tau_c", "alpha_early", "alpha_late"),
            "y": params,
        },
        "new_params_diffmah": {
            "x": ("tau_c", "a2", "alpha_late"),
            "y": params,
        },
        "alpha": {
            "x": ("alpha",),
            "y": params,
        },
        "gradients": {
            "x": am_names + grad_names_k[11],
            "y": params,
        },
        "overfitting5": {
            "x": am_names[::5],
            "y": params,
        },
        "overfitting10": {
            "x": am_names[::10],
            "y": params,
        },
    }
    datasets, _, cat_test = prepare_datasets(cat, info)

    data = {
        "multicam_ma": {
            "xy": datasets["ma"]["train"],
            "n_features": 165,
            "n_targets": len(params),
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "multicam_ma_diffmah": {
            "xy": datasets["ma_diffmah"]["train"],
            "n_features": 165,
            "n_targets": len(params),
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "multicam_params_diffmah": {
            "xy": datasets["params_diffmah"]["train"],
            "n_features": 3,
            "n_targets": len(params),
            "model": "linear",
            "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        },
        "optcam": {
            "xy": datasets["am"]["train"],
            "n_features": 100,
            "n_targets": len(params),
            "model": "mixed_cam",
            "kwargs": {
                "mass_bins": mass_bins,
                "opt_mbins": [opcam_dict[param]["mbin"] for param in params],
                "cam_orders": [opcam_dict[param]["order"] for param in params],
            },
        },
        # "multicam_diffmah_new": {
        #     "xy": datasets["diffmah_new"]["train"],
        #     "n_features": 3,
        #     "n_targets": len(params),
        #     "model": "linear",
        #     "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        # },
        # "multicam_alpha": {
        #     "xy": datasets["alpha"]["train"],
        #     "n_features": 1,
        #     "n_targets": len(params),
        #     "model": "linear",
        #     "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        # },
        # 'gradients': {
        #     'xy': datasets['gradients']['train'], 'n_features': 100 + 165, 'n_targets':
        # len(params),
        #     'model': 'linear', 'kwargs': {'to_marginal_normal':True , 'use_multicam': True},
        # },
        # "overfitting5": {
        #     "xy": datasets["overfitting5"]["train"],
        #     "n_features": 100 // 5,
        #     "n_targets": len(params),
        #     "model": "linear",
        #     "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        # },
        # "overfitting10": {
        #     "xy": datasets["overfitting10"]["train"],
        #     "n_features": 100 // 10,
        #     "n_targets": len(params),
        #     "model": "linear",
        #     "kwargs": {"to_marginal_normal": True, "use_multicam": True},
        # },
    }
    models = training_suite(data)
    metrics_data = {
        "spear": {"yrange": (0.3, 0.85)},
        # "rscatter": {"yrange": (0.3, 1.0)},
        # "sigma_ratio": {"yrange": (0.8, 1.2), "hline": 1.0},
        # "mu": {"yrange": (-0.2, 0.2), "hline": 0.0},
    }

    test_data = {
        "multicam_ma": (
            datasets["ma"]["test"][0],
            r"\rm MultiCAM $m(a)$",
            CB_COLORS[0],
            MARKS[0],
        ),
        "multicam_ma_diffmah": (
            datasets["ma_diffmah"]["test"][0],
            r"\rm MultiCAM DiffMAH $m(a)$ curves",
            CB_COLORS[2],
            MARKS[2],
        ),
        "multicam_params_diffmah": (
            datasets["params_diffmah"]["test"][0],
            r"\rm MultiCAM DiffMAH parameters",
            CB_COLORS[1],
            MARKS[1],
        ),
        "optcam": (
            datasets["am"]["test"][0],
            r"\rm CAM $a_{\rm opt}$",
            CB_COLORS[3],
            MARKS[3],
        ),
        # "overfitting5": (
        #     datasets["overfitting5"]["test"][0],
        #     r"\rm MultiCAM subsampled every 5",
        #     "g",
        #     "D",
        # ),
        # "multicam_alpha": (
        #     datasets["alpha"]["test"][0],
        #     r"\rm MultiCAM $\alpha$ only",
        #     rxplots.CB_COLORS[1],
        #     MARKS[1],
        # ),
        # "overfitting10": (
        #     datasets["overfitting10"]["test"][0],
        #     r"\rm MultiCAM subsampled every 10",
        #     "k",
        #     "*",
        # ),
        # 'gradients': (datasets['gradients']['test'][0], r"\rm MultiCAM Gradients + MAH", 'k',
        # '*'),
    }
    fig = rxplots.metrics_plot(metrics_data, test_data, models, cat_test, params=params)
    figfile = figsdir.joinpath("pred_plots.png")
    fig.savefig(figfile)


def make_covariance_am_plot():
    mahdir = root.joinpath("data", "processed", "bolshoi_m12")
    mah_data = get_mah(mahdir, cutoff_missing=0.05, cutoff_particle=0.05)
    mass_bins = mah_data["mass_bins"]
    am = mah_data["am"]
    corr_matrix = [
        [spearmanr(am[:, ii], am[:, jj])[0] for ii in range(am.shape[1])]
        for jj in range(am.shape[1])
    ]
    corr_matrix = np.array(corr_matrix).reshape(am.shape[1], am.shape[1])

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"$\rho_{\rm spearman} \left( a(m), a(m) \right)$")
    n_ticks = len(ax.get_xticklabels())
    mass_bin_labels = mass_bins[:: (len(mass_bins) + 2) // n_ticks]
    mass_bin_labels = [f"{label:.2f}" for label in mass_bin_labels]
    print(len(mass_bins), n_ticks, len(mass_bin_labels))
    assert n_ticks == len(mass_bin_labels)
    ax.set_xticklabels(mass_bin_labels)
    ax.set_yticklabels(mass_bin_labels)
    ax.matshow(corr_matrix)

    figfile = figsdir.joinpath("am_corr.png")
    fig.savefig(figfile)


def main():
    # make_correlation_mah_plots()
    # make_triangle_plots()
    make_mah_pred_plots()
    # make_inv_pred_plots()
    # make_pred_plots()
    # make_covariance_am_plot()
    pass


if __name__ == "__main__":
    main()
