'''Solar spectra of the L1 drivers: Thuillier (2003) composite, ATLAS3 and the setting solar_spectrum.'''

import logging

import numpy as np
import pytest

from hgrs.auxdata import SOLAR_SPECTRA, SolarIrradiance
from hgrs.drivers import DRIVERS
from hgrs.drivers.base import L1Driver
from hgrs.drivers.enmap import EnmapDriver
from hgrs.utils import Misc


@pytest.fixture(scope='module')
def solar():
    return SolarIrradiance()


def band_mean(F0, lo, hi):
    '''Mean irradiance over [lo, hi] nm on a 1 nm grid (the spectra have different samplings).'''
    return float(F0.interp(wl=np.arange(lo, hi + 1, 1.)).mean())


def test_thuillier2003_composite(solar):
    '''The composite covers 300-2397 nm; ATLAS3 is 1.4 % above it from 500 to 2390 nm.'''
    F0 = solar.thuillier2003
    assert float(F0.wl.min()) >= 300 and float(F0.wl.max()) == pytest.approx(2397, abs=1)
    assert band_mean(F0, 549, 551) == pytest.approx(1866, rel=0.02)
    ratio = band_mean(solar.thuillier, 500, 2390) / band_mean(F0, 500, 2390)
    assert ratio == pytest.approx(1.014, abs=0.002)


def test_tsis_over_thuillier2003_swir(solar):
    '''TSIS-1 is ~4-5 % below the Thuillier (2003) composite at 2190 nm (Sentinel-2 B12).'''
    ratio = band_mean(solar.tsis, 2100, 2280) / band_mean(solar.thuillier2003, 2100, 2280)
    assert 0.94 < ratio < 0.97


def test_get(solar):
    for name in SOLAR_SPECTRA:
        F0 = solar.get(name)
        assert F0.attrs['solar_spectrum'] == name
        assert F0.attrs['solar_spectrum_reference'] == SOLAR_SPECTRA[name]
        assert np.all(np.isfinite(F0.sel(wl=slice(400, 2390))))
    with pytest.raises(ValueError, match='unknown solar spectrum'):
        solar.get('fontenla2011')


def test_driver_default_tsis(solar):
    '''Default: TSIS-1 (products unchanged), corrected for the Earth-Sun distance.'''
    driver = L1Driver()
    F0 = driver.solar_irradiance(185)
    assert F0.attrs['solar_spectrum'] == 'tsis'
    np.testing.assert_allclose(F0.values, solar.tsis.values * Misc.earth_sun_correction(185))


def test_driver_override(solar):
    driver = L1Driver()
    driver.solar_spectrum = 'thuillier2003'
    F0 = driver.solar_irradiance(3)
    assert F0.attrs['solar_spectrum'] == 'thuillier2003'
    np.testing.assert_allclose(F0.values, solar.thuillier2003.values * Misc.earth_sun_correction(3))


def test_every_driver_declares_calibration():
    for cls in DRIVERS.values():
        attrs = cls.f0_attrs(cls.__new__(cls), 'tsis')
        assert attrs['radiance_calibration']
        assert attrs['mission_solar_spectrum']


def test_enmap_warns_on_other_spectrum(caplog):
    '''EnMAP radiance embeds Fontenla (2011): converting it with TSIS-1 is reported.'''
    driver = EnmapDriver()
    with caplog.at_level(logging.WARNING, logger='hgrs.drivers.base'):
        F0 = driver.solar_irradiance(100)
    assert 'fontenla2011' in caplog.text
    assert F0.attrs['mission_solar_spectrum'] == 'fontenla2011'
