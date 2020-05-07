from __future__ import print_function, division

import numpy as np
import binning

"""
subhaloes.py provides several utility functions for analysing a host halo's
subhaloes.

@author: Phil Mansfield
"""


class LookupTable(object):
    """ LookupTable creates a fast lookup table which can quickly associated
    subhaloes pids with host halo ids.
    """

    def __init__(self, host_ids):
        self.host_ids = host_ids
        self.host_index = np.arange(len(host_ids))

        order = np.argsort(host_ids)
        self.host_ids = self.host_ids[order]
        self.host_index = self.host_index[order]

    def lookup(self, sub_pids):
        """ lookup returns the indices of the host halos of each subhalo.
        """
        sub_index = np.searchsorted(self.host_ids, sub_pids)
        return self.host_index[sub_index]


def bin_by_host(host_ids, sub_pids):
    """ bin_by_host bins subhaloes according to their hosts. This function
    returns a list where each element is an array of indices into `sub_pids`
    the correspond to that that host's subhaloes.
    """

    table = LookupTable(host_ids)
    sub_index = table.lookup(sub_pids)
    return binning.bin_ints(sub_index, len(host_ids))


# Provided as an example of how to use bin_by_host:

def M_sub(host_ids, sub_mvir, sub_pids):
    """ M_sub returns the sum of the mass of all subhaloes of each host.
    """
    bins = bin_by_host(host_ids, sub_pids)

    M_sub = np.zeros(len(bins))
    for i in range(len(bins)):
        sub_mvir_i = sub_mvir[bins[i]]
        M_sub[i] = np.sum(sub_mvir_i)

    return M_sub


def N_sub(host_ids, sub_pids):
    bins = bin_by_host(host_ids, sub_pids)
    N_sub = np.zeros(len(bins))
    for i in range(len(M_sub)):
        sub_index = bins[i]
        N_sub[i] = len(sub_index)
    return N_sub


def mass_gap(host_mvir, host_ids, sub_mvir, sub_pids):
    bins = bin_by_host(host_ids, sub_pids)
    mass_gap = np.zeros(len(bins))
    for i in range(len(M_sub)):
        sub_index = bins[i]
        M_sub[i] = host_mivr[i] - np.max(sub_mvir[sub_index])
    return mass_gap


def run_tests():
    host_ids = np.array([100, 200, 50, -17], dtype=int)
    sub_pids = np.array([100, -17, 100, 200, -17, 100], dtype=int)
    sub_mvir = np.array([1, 2, 3, 4, 5, 6])

    # test LookupTable

    table = LookupTable(host_ids)
    sub_index = table.lookup(sub_pids)

    assert np.all(sub_index == np.array([0, 3, 0, 1, 3, 0]))

    # test bin_by_host

    bins = bin_by_host(host_ids, sub_pids)

    assert len(bins) == 4
    assert np.all(bins[0] == np.array([0, 2, 5]))
    assert np.all(bins[1] == np.array([3]))
    assert len(bins[2]) == 0
    assert np.all(bins[3] == np.array([1, 4]))

    # test M_sub

    m_sub = M_sub(host_ids, sub_mvir, sub_pids)

    assert np.all(m_sub == np.array([10, 4, 0, 7]))


if __name__ == "__main__":
    run_tests()
