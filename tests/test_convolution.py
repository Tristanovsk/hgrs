"""Spectral convolution as a sparse matrix (GenericSpecSen.convolution_matrix) and water vapor transmittance table
at the air-mass nodes of the scene (LUTTables.interp_twv): same values as the former computation."""
import numpy as np
import pytest
import xarray as xr

from hgrs.config import behavior
from hgrs.lut_tables import TWV_AIR_MASSES, LUTTables, air_mass_nodes
from hgrs.spectral_sensitivity import Gaussian, SuperGaussian, resample_to_target_raster

WL = np.arange(350., 2600., 0.25)
BANDS = np.r_[np.linspace(420, 990, 30), np.linspace(900, 2450, 40)]
FWHM = np.r_[np.full(30, 6.5), np.full(40, 10.)]


@pytest.mark.parametrize('sensor', [Gaussian(BANDS, FWHM), SuperGaussian(BANDS, FWHM)])
def test_matrix_convolution_as_loop(sensor):
    rng = np.random.default_rng(0)
    signal = xr.DataArray(rng.random((3, WL.size, 2)).astype(np.float32),
                          coords={'tcwv': [0., 1., 2.], 'wl': WL, 'b': [0, 1]}, dims=('tcwv', 'wl', 'b'))
    new = sensor.convolve(signal)
    old = resample_to_target_raster(signal, sensor.get_rsr(signal.wl), threshold=1e-6)
    assert new.dims == old.dims and new.dtype == old.dtype
    np.testing.assert_array_equal(new.wl_sensor, old.wl_sensor)
    np.testing.assert_allclose(new, old, rtol=1e-12)
    # the matrix is computed once per grid
    assert sensor.convolution_matrix(WL) is sensor.convolution_matrix(WL.copy())


def test_matrix_convolution_nan_in_band():
    sensor = Gaussian(BANDS, FWHM)
    signal = xr.DataArray(np.ones(WL.size), coords={'wl': WL}, dims='wl')
    signal[np.argmin(abs(WL - BANDS[3]))] = np.nan
    out = sensor.convolve(signal)
    assert np.isnan(out[3]) and np.isfinite(out[10])


def test_previous_behavior_keeps_the_loop(monkeypatch):
    monkeypatch.setattr(behavior, 'PREVIOUS_BEHAVIOR', True)
    sensor = Gaussian(BANDS, FWHM)
    out = sensor.convolve(xr.DataArray(np.ones(WL.size), coords={'wl': WL}, dims='wl'))
    assert out.dtype == np.float32 and '_convolution_matrices' not in sensor.__dict__


def test_air_mass_nodes():
    np.testing.assert_allclose(air_mass_nodes(2.37), [2.3, 2.4])
    np.testing.assert_allclose(air_mass_nodes(2.0), [2.0, 2.1])
    np.testing.assert_allclose(air_mass_nodes(30.), [20., 30.])
    np.testing.assert_allclose(air_mass_nodes(1.5), TWV_AIR_MASSES[:2])


@pytest.mark.parametrize('air_mass', [2.37, 2.0, 7.2, 30., 1.5])
def test_twv_table_at_the_air_mass_nodes(air_mass):
    tables = LUTTables.__new__(LUTTables)
    tables.gas_lut = xr.Dataset({'h2o': ('wl', 0.02 * np.exp(-0.5 * ((WL - 940) / 15) ** 2))}, coords={'wl': WL})
    tables._full_Twv_lut, tables._convolved_twv = None, (None, None, None)
    sensor = Gaussian(BANDS, FWHM)
    full = tables.interp_twv(BANDS, sensor).Twv
    assert full.sizes['air_mass'] == TWV_AIR_MASSES.size
    part = tables.interp_twv(BANDS, sensor, air_mass=air_mass).Twv
    assert part.sizes['air_mass'] == 2
    np.testing.assert_allclose(part.interp(air_mass=air_mass), full.interp(air_mass=air_mass), rtol=1e-12,
                               equal_nan=True)
    # cached for the same sensor and nodes
    assert tables.interp_twv(BANDS[:5], sensor, air_mass=air_mass).Twv.sizes['air_mass'] == 2
    assert tables._convolved_twv[1] == tuple(air_mass_nodes(air_mass))
