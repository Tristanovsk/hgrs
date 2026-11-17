'''Tests of hgrs.hgrs_kernel.Algo.get_omnicloudmask: with several NIR bands, flagged only where all runs agree'''
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hgrs import hgrs_kernel
from hgrs.hgrs_kernel import Algo

# fake OmniCloudMask outputs, keyed by the value of the NIR band given to it
PRED = {865: np.array([[0, 2], [1, 3]]),
        940: np.array([[2, 0], [3, 1]])}


@pytest.fixture
def fake_predict(monkeypatch):
    calls = []

    def predict(values):
        assert values.shape[0] == 3
        nir = int(values[2, 0, 0])
        calls.append(nir)
        return PRED[nir][None]

    monkeypatch.setattr(hgrs_kernel, 'predict_from_array', predict)
    return calls


def _rgnir(nirs):
    wl = [670, 550, *nirs]
    values = np.stack([np.full((2, 2), w, dtype=float) for w in wl])
    return xr.DataArray(values, dims=('wl', 'y', 'x'),
                        coords=dict(wl=wl, y=[0, 1], x=[0, 1], time=np.datetime64('2025-07-04')))


def test_single_nir_is_the_raw_prediction(fake_predict):
    mask = Algo.get_omnicloudmask(SimpleNamespace(), _rgnir([940]))
    np.testing.assert_array_equal(mask.values, PRED[940])
    assert fake_predict == [940]


def test_two_nir_flag_only_where_both_agree(fake_predict):
    mask = Algo.get_omnicloudmask(SimpleNamespace(), _rgnir([865, 940]))
    # clear where any run is clear; otherwise the class of the first NIR band
    np.testing.assert_array_equal(mask.values, [[0, 0], [1, 3]])
    assert fake_predict == [865, 940]
    assert mask.attrs['nir_wavelengths'] == [865., 940.]


def test_needs_a_nir_band(fake_predict):
    with pytest.raises(ValueError, match='at least one nir'):
        Algo.get_omnicloudmask(SimpleNamespace(), _rgnir([]))
