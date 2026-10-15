import os
import datetime as dt
import logging

import numpy as np
import xarray as xr
import h5py

from ..auxdata import SolarIrradiance
from ..utils import Reproj, Misc
from ..spectral_sensitivity import Gaussian
from ..config import SensorDescription
from .base import L1Driver

logger = logging.getLogger(__name__)


class TanagerDriver(L1Driver):
    '''Reader of the Planet Tanager basic-radiance HDF5 products.'''

    name = 'tanager'
    platform = 'Tanager'
    tiled_output = True

    def read(self, img_path: str, *, geoproject=True, reflectance_unit=True,
             drop_vars=False, parallel=False):
        """Read a Planet Tanager basic-radiance HDF5 scene.

        Return an xarray dataset using the hGRS input convention: ``Ltoa`` is
        ``(y, x, wl)``, image geometry is ``(y, x)``, and ``F0``/``fwhm`` are
        ``(wl,)``. The source HDF5 radiance is stored band-first.
        """
        l1c_path = img_path
        with h5py.File(l1c_path, 'r') as source:
            fields = source['/HDFEOS/SWATHS/HYP/Data Fields']
            geo = source['/HDFEOS/SWATHS/HYP/Geolocation Fields']
            radiance = fields['toa_radiance'][:]
            wavelengths = np.asarray(fields['toa_radiance'].attrs['wavelengths'], dtype=float)
            fwhm = np.asarray(fields['toa_radiance'].attrs['fwhm'], dtype=float)
            lat = geo['Latitude'][:]
            lon = geo['Longitude'][:]
            sun_zenith = fields['sun_zenith'][:]
            sun_azimuth = fields['sun_azimuth'][:]
            view_zenith = fields['sensor_zenith'][:]
            view_azimuth = fields['sensor_azimuth'][:]
            acquisition_times = geo['Time'][:]
            beta_cloud = fields['beta_cloud_mask'][:]
            beta_cirrus = fields['beta_cirrus_mask'][:]
            nodata = fields['nodata_pixels'][:]

        if radiance.shape[0] != wavelengths.size or fwhm.size != wavelengths.size:
            raise ValueError('Tanager radiance bands do not match wavelength metadata')
        wavelength_order = np.argsort(wavelengths)
        wavelengths = wavelengths[wavelength_order]
        fwhm = fwhm[wavelength_order]
        radiance = np.moveaxis(radiance[wavelength_order], 0, -1)

        valid_times = acquisition_times[acquisition_times != -9999]
        if not valid_times.size:
            raise ValueError('Tanager scene has no valid acquisition times')
        date = dt.datetime.utcfromtimestamp(float(np.mean(valid_times)))

        # Decode fill values before using angles or geolocation downstream.
        lat = np.where(lat == -9999, np.nan, lat)
        lon = np.where(lon == -9999, np.nan, lon)
        sun_zenith = np.where(sun_zenith == -9999, np.nan, sun_zenith)
        sun_azimuth = np.where(sun_azimuth == -9999, np.nan, sun_azimuth)
        view_zenith = np.where(view_zenith == -9999, np.nan, view_zenith)
        view_azimuth = np.where(view_azimuth == -9999, np.nan, view_azimuth)

        self.sensor_description = SensorDescription(
            name='Tanager',
            sensor_mod=Gaussian(wavelengths, fwhm),

        )
        solar_irradiance = SolarIrradiance()
        solar_distance_factor = Misc.earth_sun_correction(date.timetuple().tm_yday)
        F0 = solar_irradiance.tsis * solar_distance_factor
        F0_sensor = self.sensor_description.convolve(
            F0, solar_irradiance=True
        ).rename({'wl_sensor': 'wl'})

        data = xr.Dataset(
            data_vars={
                'Ltoa': (('y', 'x', 'wl'), radiance),
                'F0': (('wl',), F0_sensor.values),
                'fwhm': (('wl',), fwhm),
                'lon': (('y', 'x'), lon),
                'lat': (('y', 'x'), lat),
                'sza': (('y', 'x'), sun_zenith),
                'saa': (('y', 'x'), sun_azimuth),
                'vza': (('y', 'x'), view_zenith),
                'vaa': (('y', 'x'), view_azimuth),
                'raa': (('y', 'x'), (sun_azimuth - view_azimuth) % 360),
                'nodata_pixels': (('y', 'x'), nodata),
                'beta_cloud_mask': (('y', 'x'), beta_cloud),
                'beta_cirrus_mask': (('y', 'x'), beta_cirrus),
            },
            coords={
                'x': np.arange(lon.shape[1]),
                'y': np.arange(lon.shape[0]),
                'wl': wavelengths,
                'time': date,
            },
            attrs={
                'description': 'Tanager basic radiance scene',
                'platform': 'Tanager',
                'L1C_product_name': os.path.basename(l1c_path),
                'acquisition_date': date.isoformat(),
            },
        )
        data['Ltoa'] = data.Ltoa.where(data.Ltoa >= 0)
        if reflectance_unit:
            data['Rtoa'] = np.pi * data.Ltoa / (
                data.F0 * np.cos(np.radians(data.sza))
            )
            valid = ((data.beta_cloud_mask == 0) &
                     (data.beta_cirrus_mask == 0) &
                     (data.nodata_pixels == 0))
            data['Rtoa'] = data.Rtoa.where(valid)
            if drop_vars:
                data = data.drop_vars('Ltoa')

        data.F0.attrs.update(
            unit='mW/m2/nm',
            definition='Solar irradiance corrected for Sun-Earth distance',
        )
        if 'Ltoa' in data:
            data.Ltoa.attrs.update(unit='mW/m2/sr/nm', definition='Top-of-atmosphere radiance')
        data.fwhm.attrs.update(unit='nm', definition='Full Width at Half Maximum')

        source_georeferencing = self.source_georeferencing(data)
        if geoproject:
            native_resolution_m = Reproj.ground_sampling_resolution_m(lon, lat)
            data.attrs['native_ground_sampling_m'] = native_resolution_m
            data = Reproj().regridding(
                data,
                output_resolution_m=native_resolution_m,
                parallel=parallel,
            )
        self.scene_metadata = self.make_scene_metadata(
            data,
            platform='Tanager',
            source_georeferencing=source_georeferencing,
            source_paths={'l1c': l1c_path},
        )
        return data
