"""
We want to test the following:
1. check correct value with log and without log is returned.
2. Check correct text is returned in different cases.
3. Check that modifying a parameter via modifiers works.
Later:

"""

import pytest
from relaxed.frames import params
import numpy as np


@pytest.fixture
def small_bolshoi_cat():
    """
    Return a small bolshoi catalog (one of the high z) for testing.
    :return:
    """
    from relaxed.frames import catalogs
    from relaxed.utils import const
    path_to_cat = const.joinpath("data/bolshoi/...")
    hcat = catalogs.HaloCatalog(path_to_cat, "Bolshoi")
    return hcat


class TestParams(object):
    def __init__(self):
        self.hcat = None

    def setup(self, small_bolshoi_cat):
        self.hcat = small_bolshoi_cat


@pytest.mark.parametrize("param_name", ['mvir', 'rvir', 'q'])
def test_log(bhcat, param_name):
    bhcat.set_cat(use_generator=False)
    cat = bhcat.cat

    param_mvir = params.Param("mvir", log=False)

    assert np.allclose(param_mvir.get_values(cat), cat['mvir'])
    assert not np.allclose(param_mvir.get_values(cat), np.log10(cat['mvir']))

    param_mvir_log = params.Param("mvir", log=True)
    assert not np.allclose(param_mvir_log.get_values(cat), cat['mvir'])
    assert np.allclose(param_mvir_log.get_values(cat), np.log10(cat['mvir']))


def test_text(bhcat):
    pass

