import numpy as np

from bin.catalog_pipeline import get_central_subhaloes


def test_was_central():
    curr_ids = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13])
    prev_pids = np.array([-1, 1, 2, -1, 4, -1])
    prev_dfids = np.array([5, 10, 11, 15, 35, 47])
    curr_dfids = np.array([9, 10, 14, 1, 44, 4, 27, 46, 46])
    curr_pids = np.array([-1, -1, 8, -1, 8, 11, -1, 7, 7])

    sub_keep = get_central_subhaloes(prev_pids, prev_dfids, curr_ids, curr_pids, curr_dfids)
    np.testing.assert_equal(sub_keep, np.array([0, 0, 1, 0, 0, 1, 0, 1, 1]).astype(bool))
