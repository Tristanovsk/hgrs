"""Coupled water vapor fit: non-negative surface term, bounds of the path scale, offset of the saturated bands."""
import numpy as np
import pytest
import xarray as xr
from scipy.optimize import least_squares

from hgrs.config.atmospheric_correction import WaterVaporConfig
from hgrs.hgrs_kernel import WaterVapor, _fit_water_vapor_coupled_block, _water_vapor_residuals_coupled

WL = np.linspace(0.931, 1.060, 12)
W_GRID = np.array([0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60.])
# synthetic absorption band centred at 945 nm; the path term sees a shorter path (shallower band)
SIGMA = 0.02 * np.exp(-0.5 * ((WL - 0.945) / 0.015) ** 2)
TWV = np.exp(-np.outer(W_GRID, SIGMA) * 2.5)
TATM = np.exp(-np.outer(W_GRID, SIGMA) * 1.2)
PATH = 0.006 * (WL / 1.0) ** -2


def simulate(w, k, s0, s1, dn=0.):
    surface = s0 + (s1 - s0) * (WL - WL[0]) / (WL[-1] - WL[0])
    tw = np.array([np.interp(w, W_GRID, TWV[:, j]) for j in range(WL.size)])
    ta = np.array([np.interp(w, W_GRID, TATM[:, j]) for j in range(WL.size)])
    return ta * k * PATH * WL ** -dn + tw * surface


def fit(y, surface, limits=(0.0, 5.0), w0=10., path_slope='fixed'):
    spectra = y[:, None, None]
    settings = dict(x0=[w0, 1., 0., 0.], tcwv_grid=W_GRID, Twv_table=TWV, Tatm_table=TATM, path=PATH, wl=WL,
                    pixel_threshold=0, regularization=None, surface=surface, path_scale_limits=limits,
                    path_slope=path_slope, path_slope_limits=(-4., 4.))
    return _fit_water_vapor_coupled_block((spectra, None, settings))[0, 0]


@pytest.mark.parametrize('w, k, s0, s1', [(5., 1., 0.002, 0.001), (25., 1.3, 0.0, 0.0005), (40., 0.8, 0.01, 0.006)])
def test_non_negative_surface_recovers_the_column(w, k, s0, s1):
    result = fit(simulate(w, k, s0, s1), 'non_negative', limits=(0.5, 2.0))
    assert result[0] == pytest.approx(w, rel=1e-3)
    assert result[1] == pytest.approx(k, rel=1e-3)
    assert result[2] == pytest.approx(s1, abs=1e-6)


@pytest.mark.parametrize('dn', [-2., 1.5])
def test_free_path_slope(dn):
    # path reflectance steeper or flatter than the LUT: with a fixed shape the surface term corrects the slope
    # and W is biased; with path_slope: free, W is recovered
    y = simulate(20., 1., 0.0005, 0.0005, dn)
    fixed = fit(y, 'non_negative', limits=(0.5, 2.0))
    free = fit(y, 'non_negative', limits=(0.5, 2.0), path_slope='free')
    assert free[0] == pytest.approx(20., rel=1e-3)
    assert abs(fixed[0] - 20.) > 10 * abs(free[0] - 20.)


def test_path_scale_limits_are_bounds():
    # data with twice the LUT path reflectance: k stops at the upper bound
    result = fit(simulate(10., 2.0, 0.001, 0.001), 'non_negative', limits=(0.5, 1.5))
    assert result[1] == pytest.approx(1.5)


def test_linear_surface_default_unchanged():
    # surface: linear with the default path_scale_limits is the fit of 2.0.0 (same call of least_squares)
    y = simulate(10., 1., 0.002, 0.001)
    result = fit(y, 'linear')
    ref = least_squares(_water_vapor_residuals_coupled, [10., 1., 0., 0.],
                        args=(W_GRID, TWV, TATM, PATH, WL, y), bounds=([0, 0, -10, -1], [60, 5, 1, 1]),
                        diff_step=1e-2, xtol=1e-2, ftol=1e-2, max_nfev=40).x
    np.testing.assert_array_equal(result[:3], ref[[0, 1, 3]])


def test_config_validation():
    assert WaterVaporConfig().surface == 'linear'
    assert WaterVaporConfig().path_scale_limits == (0.0, 5.0)
    assert WaterVaporConfig.from_mapping(dict(path_scale_limits=[0.5, 2])).path_scale_limits == (0.5, 2)
    with pytest.raises(ValueError, match='surface'):
        WaterVaporConfig(surface='positive')
    with pytest.raises(ValueError, match='offset'):
        WaterVaporConfig(offset='fit')
    with pytest.raises(ValueError, match='path_scale_limits'):
        WaterVaporConfig(path_scale_limits=(2.0, 1.0))
    with pytest.raises(ValueError, match='requires surface'):
        WaterVaporConfig(path_slope='free')
    assert WaterVaporConfig(surface='non_negative', path_slope='free').path_slope_limits == (-4.0, 4.0)


def test_offset_of_the_saturated_bands():
    wl = np.array([940., 1000., 1355., 1367., 1379., 1391., 1460.])
    data = np.zeros((wl.size, 2, 3))
    data[3:6] = np.array([0.001, 0.002, 0.003])[:, None, None]
    data[3:6, 0, 0] = np.nan
    raster = xr.Dataset(dict(Rtoa=(('wl', 'y', 'x'), data)), coords=dict(wl=wl, y=[0, 1], x=[0, 1, 2]))
    wv = WaterVapor.__new__(WaterVapor)
    wv.raster, wv.variable = raster, 'Rtoa'
    wv.parameters = WaterVaporConfig(offset='saturated_bands')
    offset = wv.offset()
    assert offset.shape == (2, 3)
    assert offset[0, 0] == 0.
    np.testing.assert_allclose(offset[1], 0.002)
    wv.parameters = WaterVaporConfig()
    assert wv.offset() is None
