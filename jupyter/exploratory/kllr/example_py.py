from kllr import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

print(sys.path)

df = pd.read_csv("./data/TNG300_Halos.csv")

print(df.columns)

df = df[df.M200 > 13.5]
Colors = ["#FF7F0E", "mediumseagreen", "mediumpurple", "steelblue"]
# Colors = ['indianred', 'orange', 'steelblue']

# simple regression
data, ax = Plot_Fit(
    df,
    "M200",
    "MStar_BCG100",
    show_data=True,
    ax=None,
    xlog=True,
    kernel_width=0.4,
    ylog=False,
    labels=[r"$M200\,\,[M_\odot]$", r"$M_{\rm BCG, 100kpc}\,\,[M_\odot]$"],
)
plt.show()

# regressing MStar_BCG100 against M200
# splitting on formation time.
# xlog/ylog specifies if the quantities in x/y axis are log quantities.
# split_bins: specifies how many bins it should use it split on equal size quantiles.
data, ax = Plot_Fit_Split(
    df,
    "M200",
    "MStar_BCG100",
    "z_form",
    show_data=True,
    xlog=True,
    ylog=False,
    split_bins=4,
    color=Colors,
    kernel_width=0.4,
    labels=[
        r"$M200\,\,[M_\odot]$",
        r"$M_{\rm BCG, 100kpc}\,\,[M_\odot]$",
        r"$z_{\rm form}$",
    ],
)
plt.show()


# regressing  MStar_BCG100 against z_form
# splitting on halo mass
# you can define the boundaries of your split variable instead of splitting them into equal quantiles.
data, ax = Plot_Fit_Split(
    df,
    "z_form",
    "MStar_BCG100",
    "M200",
    show_data=True,
    xlog=False,
    ylog=False,
    split_bins=[13.5, 14, 14.5, 15.5],
    kernel_width=0.4,
    color=["indianred", "orange", "steelblue"],
    labels=[
        r"$z_{\rm form}$",
        r"$M_{\rm BCG, 100kpc}\,\,[M_\odot]$",
        r"$\log(M200\,\,[M_\odot])$",
    ],
)
plt.show()


Plot_Fit_Params(
    df,
    "M200",
    "MStar_BCG100",
    kernel_width=0.4,
    nBootstrap=200,
    xlog=True,
    labels=[r"$M200\,\,[M_\odot]$", r"$M_{\rm BCG, 100kpc}$"],
)
plt.show()
#


Plot_Fit_Params_Split(
    df,
    "M200",
    "MStar_BCG100",
    "z_form",
    split_bins=3,
    kernel_width=0.4,
    xlog=True,
    nBootstrap=200,
    labels=[r"$M200\,\,[M_\odot]$", r"$M_{\rm BCG, 100kpc}$", r"$z_{\rm form}$"],
)
plt.show()


# plot the conditional correlation/covariance matrix of a set of parameters
# Output_mode defined whether it should be the correlation or covariance matrix.
ax = Plot_Cov_Corr_Matrix(
    df,
    "M200",
    ["MGas", "MGas_T", "c200c"],
    kernel_width=0.4,
    nBootstrap=200,
    labels=[r"$M_{200}\,\,[M_\odot]$", r"$M_{\rm gas}$", r"$T_X$", r"$c$"],
    Output_mode="corr",
)
plt.show()

ax = Plot_Cov_Corr_Matrix(
    df, "M200", ["MGas", "MStar"], nBootstrap=50, kernel_width=0.4, Output_mode="corr"
)
plt.show()


# plot the conditional correlation/covariance matrix of a set of parameters
# it additionally split on a third quantity (here z_form).
# Output_mode defined whether it should be the correlation or covariance matrix.
ax = Plot_Cov_Corr_Matrix_Split(
    df,
    "M200",
    ["MGas", "MGas_T", "MStar_BCG100", "c200c"],
    "z_form",
    split_bins=3,
    color=["indianred", "orange", "steelblue"],
    nBootstrap=50,
    kernel_width=0.4,
    labels=[
        r"$M_{200}\,\,[M_\odot]$",
        r"$M_{\rm gas}$",
        r"$T_X$",
        r"$M_{\rm BCG, 100kpc}$",
        r"$c$",
        r"$z_{\rm form}$",
        r"$z_{\rm form}$",
    ],
    Output_mode="corr",
)
plt.show()


# in case you want to study resoduals to make sure they follow the log-normal / normal model you assumed
data, ax = Plot_Residual(df, "M200", "MGas", nBootstrap=200, kernel_width=0.4)
plt.show()

data, ax = Plot_Residual_Split(
    df,
    "M200",
    "MGas",
    "z_form",
    split_bins=[0.0, 0.5, 1.0, 2.0],
    labels=[r"$\ln(M_{\rm gas})$", r"$z_{\rm form}$"],
    nBootstrap=200,
    kernel_width=0.4,
    color=["indianred", "orange", "steelblue"],
)
plt.show()


# for split_bins you can define the boundary of the bin or specify how many bins it should
# use and it automatically compute the equal size quantiles and define the bin boundries.

# you can specify which colors it should use for each line or it automatically assign a color to them


# you can define the size of your kernel function with kernel_width. It has the same unit as your x variable.
