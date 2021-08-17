import numpy as np
from astropy.table import Table
from pminh import minh

from relaxed.halo_filters import intersect
from relaxed.subhaloes import quantities


def create_subhalo_cat(host_ids, minh_file, threshold=1.0 / 100):
    """

    Args:
        * minh_file: complete minh catalog and must be read in blocks
        * host_ids corresponds only to host haloes w/ upid == -1
        * threshold: min mass of subhalo considered in `f_sub` calculation relative to
            host halo mass.

    Returns:
        Astropy table with columns 'id', 'mvir', 'f_sub', 'm2' where

        * f_sub = ratio of sum of all subhaloes to mass of host halo.
        * m2 = ratio of most massive subhalo mass to host mass
    """
    assert isinstance(host_ids, np.ndarray)
    assert np.all(np.sort(host_ids) == host_ids)

    # will fill out as we read the blocks.
    subcat_names = ["id", "mvir", "f_sub", "m2"]
    subcat_data = [host_ids, *[np.zeros(len(host_ids)) for _ in range(len(subcat_names) - 1)]]
    subcat = Table(names=subcat_names, data=subcat_data)
    subcat.sort("id")

    with minh.open(minh_file) as mcat:
        assert mcat.blocks == 1, "Only 1 block supported"
        b = 0
        names = ["id", "upid", "mvir"]
        ids, upid, mvir = mcat.block(b, names)
        table = Table(names=names, data=[ids, upid, mvir])
        table.sort("id")

        # first we calculate host_mvir
        keep = intersect(np.array(table["id"]), host_ids)
        indx_ok = intersect(host_ids, np.array(table[keep]["id"]))
        subcat["mvir"][indx_ok] = table[keep]["mvir"]
        # FIXME: Assertion only true if 1 block, code above is general.
        assert np.all(subcat["mvir"] > 0)

        # need to contain only ids of host_ids for it to work.
        sub_pids = upid[upid != -1]
        sub_mvir = mvir[upid != -1]

        # first calculate total subhalo mass
        # FIXME: For >1 block need to replace `host_ids` below.
        subcat["f_sub"] += quantities.m_sub(
            host_ids, subcat["mvir"], sub_pids, sub_mvir, threshold=threshold
        )

        # and most massive subhalo mass
        m2_curr = quantities.m2_sub(host_ids, sub_pids, sub_mvir).reshape(-1, 1)
        m2_prev = np.array(subcat["m2"]).reshape(-1, 1)
        m2 = np.hstack([m2_curr, m2_prev]).max(axis=1)
        subcat["m2"] = m2

    # finally ratio
    subcat["f_sub"] = subcat["f_sub"] / subcat["mvir"]
    subcat["m2"] = subcat["m2"] / subcat["mvir"]
    return subcat
