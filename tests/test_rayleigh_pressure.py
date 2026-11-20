"""Pressure scaling of the Rayleigh scattering of the path reflectance (hgrs_kernel.rayleigh_pressure_delta,
Aerosol.prepare_lut)."""
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hgrs import hgrs_kernel
from hgrs.config import behavior
from hgrs.hgrs_kernel import LUT_PRESSURE, Aerosol, rayleigh_pressure_delta

WL_NODES = np.array([400., 500., 600., 700., 800., 1000., 1600., 2200.])
#: Rayleigh optical thickness (~ lambda^-4.08, Bodhaine-like) at the nodes
TAU = 0.36 * (WL_NODES / 400.) ** -4.08


def test_lut_pressure_is_that_of_the_osoaa_runs():
    assert LUT_PRESSURE == 1015.2


def test_rayleigh_pressure_delta():
    """Nil at the pressure of the LUT; R_ray [(P/P0)^D - 1] with D the exponent of a power law R ~ tau^D."""
    Rray = xr.DataArray(0.4 * TAU ** 0.9, coords={'wl': WL_NODES}, dims='wl')
    wl = np.array([450., 865.])
    np.testing.assert_allclose(rayleigh_pressure_delta(Rray, TAU, LUT_PRESSURE, wl), 0., atol=1e-15)
    d = rayleigh_pressure_delta(Rray, TAU, 995.2, wl)
    R_wl = 0.4 * (0.36 * (wl / 400.) ** -4.08) ** 0.9
    np.testing.assert_allclose(d, R_wl * ((995.2 / LUT_PRESSURE) ** 0.9 - 1), rtol=2e-3)


def _aerosol(pressure):
    """Aerosol retrieval object with a synthetic LUT: Rayleigh reflectance + aerosol term linear in aot_ref."""
    aero = Aerosol.__new__(Aerosol)
    sza, vza, azi = [20., 40., 60.], [0., 10., 20.], [80., 90., 100.]
    aot_ref = [0., 0.1, 0.2, 0.5, 1.]
    ray = 0.4 * TAU ** 0.9
    I = (ray[None, :] + 0.05 * np.asarray(aot_ref)[:, None])
    I = np.broadcast_to(I[None, :, :, None, None, None], (1, 5, len(WL_NODES), 3, 3, 3)).copy()
    I = I * np.cos(np.radians(np.asarray(sza)))[None, None, None, :, None, None]
    coords = {'model': ['M'], 'aot_ref': aot_ref, 'wl': WL_NODES, 'sza': sza, 'vza': vza, 'azi': azi}
    aero.aero_lut = xr.Dataset({
        'I': (('model', 'aot_ref', 'wl', 'sza', 'vza', 'azi'), I),
        'aot': (('model', 'aot_ref', 'wl'), np.broadcast_to(np.asarray(aot_ref)[None, :, None], (1, 5, 8)).copy())},
        coords=coords)
    aero.aerosol_model = 'M'
    aero.auxdata = SimpleNamespace(rot=xr.DataArray(TAU, coords={'wl': WL_NODES}, dims='wl') * 1013.25 / 1015.2,
                                   sunglint_eps=xr.DataArray(np.ones(8), coords={'wl': WL_NODES}, dims='wl'),
                                   pressure_rot_ref=1013.25)
    aero.sza, aero.vza, aero.raa_lut, aero.air_mass = 40., 10., 90., 2.3
    aero.pressure = pressure
    aero.wl_sunglint = [1600.]
    return aero


@pytest.mark.parametrize('previous', [False, True])
def test_prepare_lut(previous, monkeypatch):
    """prepare_lut adds dRray to the path reflectance of all the aot_ref (current behavior only)."""
    monkeypatch.setattr(behavior, 'PREVIOUS_BEHAVIOR', previous)
    wl = np.array([450., 865., 1600.])
    ref, low = _aerosol(LUT_PRESSURE), _aerosol(995.2)
    ref.prepare_lut(wl)
    low.prepare_lut(wl)
    diff = (low.Rtoa_lut - ref.Rtoa_lut).values
    if previous:
        np.testing.assert_allclose(diff, 0., atol=1e-15)
        assert not hasattr(low, 'dRray')
    else:
        np.testing.assert_allclose(ref.dRray, 0., atol=1e-15)
        # the same change for all the aot_ref, negative at lower pressure, larger in the blue
        assert np.allclose(diff, diff[0][None, :])
        assert (diff < 0).all() and diff[0, 0] < diff[0, 1] < diff[0, 2]
        np.testing.assert_allclose(low.model.Rdiff_table - ref.model.Rdiff_table, diff)


@pytest.mark.parametrize('previous', [False, True])
@pytest.mark.parametrize('pressure', [LUT_PRESSURE, 995.2])
def test_water_vapor_path_reflectance_as_aerosol_fit(previous, pressure, monkeypatch):
    """Path term of the coupled water vapor model: the path reflectance of the aerosol fit (log-log wavelength
    interpolation and pressure term); quadratic in linear space without the pressure term (former behavior)."""
    monkeypatch.setattr(behavior, 'PREVIOUS_BEHAVIOR', previous)
    aero = _aerosol(pressure)
    prod = SimpleNamespace(aero_lut=aero.aero_lut, aerosol_parameters=SimpleNamespace(aerosol_model='M'),
                           sza_mean=aero.sza, vza_mean=aero.vza, raa_mean=(180 - aero.raa_lut) % 360,
                           auxdata=aero.auxdata, pressure=pressure)
    wl = np.linspace(931., 1060., 12)
    wv = SimpleNamespace(wl=xr.DataArray(wl, coords={'wl': wl}, dims='wl'))
    aot = 0.1
    path = hgrs_kernel.WaterVapor.path_reflectance_first_guess(wv, prod, aot)
    I = aero.aero_lut.I.sel(model='M').interp(vza=aero.vza, azi=aero.raa_lut).interp(sza=aero.sza, method='quadratic')
    linear = (I.interp(aot_ref=aot).interp(wl=wl, method='quadratic') / np.cos(np.radians(aero.sza))).values
    if previous:
        np.testing.assert_allclose(path, linear, rtol=1e-12)
    else:
        aero.wl_sunglint = [wl[-1]]
        aero.prepare_lut(wl)
        # prepare_lut interpolates in wavelength (log-log) before aot_ref, the water vapor path after: the
        # log-log interpolation of a sum does not commute with that in aot_ref (1e-6 here)
        np.testing.assert_allclose(path, aero.Rtoa_lut.interp(aot_ref=aot).values, rtol=1e-5)
        assert np.abs(path / linear - 1).max() > 1e-3
