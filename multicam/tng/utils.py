"""Several utility functions to read data from TNG catalogs."""
import pickle

import h5py
import numpy as np
import pandas as pd
from scipy import spatial

from multicam.parameters import get_vvir

SNAPS = np.arange(0, 100, 1)
TNG_H = 0.6774  # from website


def convert_tng_mass(gmass):
    """Convert TNG mass to log10(Msun)."""
    # TNG units are 1e10 Msun / h; https://www.tng-project.org/data/docs/specifications
    # return in units of log10(Msun)
    # robust to 0 mass
    return np.where(gmass > 0, np.log10(gmass * 1e10 / TNG_H), 0)


def get_vmax_over_vvir(cat: pd.DataFrame):
    """Compute vmax / vvir from catalog columns."""
    # compute vvir and create new column

    # ensure units of mvir is in units of Msun / h
    mvir = 10 ** cat["Mvir"].values / TNG_H  # og units: log10(msun)
    rvir = cat["Rvir"].values / TNG_H  # og units: kpc
    vvir = get_vvir(rvir, mvir)

    return cat["Vmax_DM"] / vvir


def _reverse_trees(trees):
    """Reverse each entry in trees so that order is from early to late times."""
    for tree in trees:
        for key in tree.keys():
            if key not in ["Number", "ChunkNumber", "TreeID"]:
                tree[key] = tree[key][::-1]
    return trees


def read_trees(trees_file: str):
    """Read in the trees file and convert masses to log10(M/Msun)."""
    with open(trees_file, "rb") as pickle_file:
        _trees = pickle.load(pickle_file)
        trees = _reverse_trees(_trees)
        for tree in trees:
            for k in tree.keys():
                if "Mass" in k or "_M_" in k:
                    tree[k] = convert_tng_mass(tree[k])
    return trees


def get_msmhmr(cat, gmass, mass_bin=(12.8, 13.1), n_bins=11):
    """Compute mean stellar mass to halo mass relation and deviation."""
    mvir = gmass

    ratio = np.log10(10 ** cat["Mstar_30pkpc"] / 10**mvir)
    ratio = ratio.values

    # compute mean ratio in bins of mvir
    bins = np.linspace(mass_bin[0], mass_bin[1], n_bins)
    mean_ratio_per_bin = np.zeros(len(bins) - 1)
    for ii in range(len(bins) - 1):
        idx = np.where((mvir > bins[ii]) & (mvir < bins[ii + 1]))[0]
        mean_ratio_per_bin[ii] = np.mean(ratio[idx])

    middle_point_of_bins = (bins[1:] + bins[:-1]) / 2

    m, b = np.polyfit(middle_point_of_bins, mean_ratio_per_bin, 1)

    # finally, calculate deviation from mean log ratio
    #  want \Delta Log ( M_star )
    m_star_dev = cat["Mstar_30pkpc"] - np.log10(10 ** (m * mvir + b) * 10**mvir)
    m_star_dev = m_star_dev.values

    return m_star_dev, (m, b)


def get_color(color_file: str, cat: pd.DataFrame):
    """Read in color file and return dataframe with colors (in order of catalog)."""
    f = h5py.File(color_file, "r")

    colnames = (
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "wfc_acs_f606w",
        "des_y",
        "jwst_f150w",
    )
    arr = f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:]

    # average over projections
    arr = np.mean(arr, axis=-1)
    df_color = pd.DataFrame(arr, columns=colnames)

    color_ids = f["subhaloIDs"][:]

    f.close()

    df_color = df_color.iloc[np.where(np.isin(color_ids, cat["SubhaloID"].values))[0]]

    assert all(df_color.index.values == cat["SubhaloID"].values)

    return df_color


def setup_mah_and_cat(
    trees_file: str,
    present_snapfile: str,
    metadata_file: str,
    snaps: np.array,
    mass_bin=(12.8, 13.1),
    scale_cut=0.185,  # corresponds to original paper
):
    """Read in trees and present-day catalog, and return mah and catalog."""

    # get scales and redshifts
    metadata = pd.read_csv(metadata_file)
    zs = metadata["Redshift"].values
    scales = 1 / (1 + zs)
    assert len(snaps) == len(scales)
    n_snaps = len(snaps)

    # read trees and present-day catalog
    trees = read_trees(trees_file)
    cat = pd.read_hdf(present_snapfile)

    # select trees in mass bin that have a snapshot at z=0
    trees = [
        t
        for t in trees
        if 99 in t["SnapNum"]
        and t["Group_M_TopHat200"][-1] > mass_bin[0]
        and t["Group_M_TopHat200"][-1] < mass_bin[1]
    ]

    # get mah from trees, and convert to correct units
    mah = np.zeros((len(trees), n_snaps))
    for ii, t in enumerate(trees):
        mah_t = np.zeros(n_snaps) * np.nan
        t_snaps = t["SnapNum"]
        gmass = t["Group_M_TopHat200"]
        mah_t[t_snaps] = gmass

        # linearly interpolate nan values
        mah_t = pd.Series(mah_t)
        mah_t = mah_t.interpolate(method="linear", limit_direction="forward", axis=0)
        mah[ii] = mah_t.values

    idx = np.where(scales > scale_cut)[0][0]
    snaps = snaps[idx:]
    scales = scales[idx:]
    mah = mah[:, idx:]

    # remove haloes with nans and hope not too many
    kp_idx = np.where(np.isnan(mah).sum(axis=1) == 0)[0]
    mah = mah[kp_idx]
    assert np.isnan(mah).sum() < 5

    # turn mah in m_peak
    # which is the normalized cumulative maximum
    Mpeak = np.fmax.accumulate(10**mah, axis=1)
    m_peak = Mpeak / Mpeak[:, -1][:, None]

    halo_idx = np.array([t["IndexInHaloTable"][-1] for t in trees])[kp_idx]
    cat = cat.iloc[halo_idx]

    # sort everything by 'SubhaloID' for good measure
    subhalo_id = cat["SubhaloID"].values
    sort_idx = np.argsort(subhalo_id)
    mah = mah[sort_idx]
    m_peak = m_peak[sort_idx]
    halo_idx = halo_idx[sort_idx]
    cat = cat.sort_values(by="SubhaloID")

    # post processing quantities
    cat["Vmax_DM/V_vir_DM"] = get_vmax_over_vvir(cat)

    return {
        "present_cat": cat,
        "halo_idx": halo_idx,
        "mah": mah,
        "m_peak": m_peak,
        "gmass": mah[:, -1],
        "z": zs,
        "snaps": snaps,
        "scales": scales,
    }


def match_dm_and_hydro_cat(dcat: pd.DataFrame, cat: pd.DataFrame):
    """Matching using KD tree on halo positions."""

    # get positions
    pos = np.array(cat[["pos_x", "pos_y", "pos_z"]])
    dpos = np.array(dcat[["pos_x", "pos_y", "pos_z"]])

    # kdtree
    tree = spatial.KDTree(pos)
    dtree = spatial.KDTree(dpos)

    # find the nearest neighbor in the other catalog
    d, indx = tree.query(dpos)
    _, dindx = dtree.query(pos)

    # keep only bijectively matched haloes
    keep = []
    for ii in range(len(cat)):
        if ii == indx[dindx[ii]]:
            keep.append(ii)
    keep = np.array(keep)

    dkeep = []
    for ii in range(len(dcat)):
        if ii == dindx[indx[ii]]:
            dkeep.append(ii)
    dkeep = np.array(dkeep)

    return cat.iloc[keep], dcat.iloc[dkeep], d[keep], keep
