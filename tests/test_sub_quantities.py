import numpy as np

from relaxed.subhaloes.quantities import LookupTable, bin_by_host, m2_sub, m_sub


def test_sub_quantities():
    host_mvir = np.array([10, 5, 10, 8])
    host_ids = np.array([100, 200, 50, -17], dtype=int)
    sub_pids = np.array([100, -17, 100, 200, -17, 100, 75, 300, -20], dtype=int)
    sub_mvir = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

    # test LookupTable

    table = LookupTable(host_ids)
    sub_index = table.lookup(sub_pids)

    assert np.all(sub_index == np.array([0, 3, 0, 1, 3, 0, -1, -1, -1]))

    # test bin_by_host

    bins = bin_by_host(host_ids, sub_pids)

    assert len(bins) == 4
    assert np.all(bins[0] == np.array([0, 2, 5]))
    assert np.all(bins[1] == np.array([3]))
    assert len(bins[2]) == 0
    assert np.all(bins[3] == np.array([1, 4]))

    # test M_sub

    M_sub = m_sub(host_ids, host_mvir, sub_pids, sub_mvir, threshold=1 / 4)
    m2 = m2_sub(host_ids, sub_pids, sub_mvir)

    np.testing.assert_equal(M_sub, np.array([9, 4, 0, 5]))
    np.testing.assert_equal(m2, np.array([6, 4, 0, 5]))
