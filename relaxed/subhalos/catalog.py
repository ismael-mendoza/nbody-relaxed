import numpy as np
from . import subhalo
import astropy
from astropy.table import Table


def extract_subhalo(host_cat, minh_cat):
    # TODO: change this function to accommodate lower mass host halos.
    # now we also want to add subhalo fraction and we follow Phil's lead

    host_ids = host_cat["id"]
    host_mvir = host_cat["mvir"]
    M_sub_sum = np.zeros(len(host_mvir))

    for b in range(minh_cat.blocks):
        upid, mvir = minh_cat.block(b, ["upid", "mvir"])

        # need to contain only ids of host_ids for it to work.
        sub_pids = upid[upid != -1]
        sub_mvir = mvir[upid != -1]
        M_sub_sum += subhalo.m_sub(host_ids, sub_pids, sub_mvir)

    f_sub = M_sub_sum / host_mvir  # subhalo mass fraction.
    subhalo_cat = Table(data=[host_ids, f_sub], names=["id", "f_sub"])

    return subhalo_cat


def add_subhalo_info(host_hcat, minh_cat):
    # mcat is complete
    # host_hcat contains only host haloes w/ upid == -1
    host_cat = host_hcat.cat
    assert np.all(host_cat["upid"] == -1), "Needs to be a host catalog"
    assert "mvir" in host_cat.colnames and "ids" in host_cat.colnames
    subhalo_cat = extract_subhalo(host_cat, minh_cat)
    fcat = astropy.table.join(host_cat, subhalo_cat, keys="id")
    host_hcat.cat = fcat
    return host_hcat
