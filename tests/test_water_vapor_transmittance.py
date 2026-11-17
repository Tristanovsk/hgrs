'''Tests of hgrs.hgrs_kernel.Algo.get_wv_transmittance_raster on a synthetic water vapor LUT'''
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hgrs.hgrs_kernel import Algo


def _product():
    tcwv = np.array([0., 10., 20., 40.])
    air_mass = np.array([2., 3.])
    wl = np.array([900., 940.])
    Twv = xr.DataArray(np.exp(-0.01 * tcwv[:, None, None] * air_mass[None, :, None] * np.array([0.1, 1.])),
                       coords=dict(tcwv=tcwv, air_mass=air_mass, wl=wl), dims=('tcwv', 'air_mass', 'wl'))
    return SimpleNamespace(Twv_lut=xr.Dataset(dict(Twv=Twv)), air_mass_mean=2.5)


def _retrieval(values):
    return xr.Dataset(dict(tcwv=(('y', 'x'), np.array(values, dtype=float))))


def test_all_nan_retrieval_raises_clear_error():
    prod = _product()
    with pytest.raises(ValueError, match='no valid water vapor retrieval'):
        Algo.get_wv_transmittance_raster(prod, _retrieval([[np.nan, np.nan], [np.nan, np.nan]]))


def test_partial_nan_retrieval_gives_raster():
    prod = _product()
    Algo.get_wv_transmittance_raster(prod, _retrieval([[15., np.nan], [30., 15.]]))
    T = prod.Twv_raster.transpose('y', 'x', 'wl')
    assert T.shape == (2, 2, 2)
    assert np.isfinite(T.values[0, 0]).all() and np.isfinite(T.values[1, 0]).all()
    assert (T.values[1, 0] < T.values[0, 0]).all()     # more water vapor, lower transmittance
