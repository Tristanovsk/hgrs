"""Reader for NASA EMIT L1B calibrated-radiance NetCDF products."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr

from . import Misc, Reproj, SolarIrradiance
from .config import SceneMetadata, SensorDescription
from .spectral_sensitivity import Gaussian


class EmitDriver:
    """Read EMIT L1B radiance into hGRS's sensor scene contract.

    Geometry is not supplied in EMIT L1B_RAD; callers must provide scene solar
    azimuth/elevation and a scene-constant viewing zenith/azimuth. The default
    grid is the source swath. ``geoproject=True`` applies hGRS geographic
    bilinear interpolation; ``ortho=True`` uses the product's GLT grid instead.
    """

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.scene_metadata = None
        self.sensor_description = None

    def read(self, sun_azimuth, sun_elevation, view_azimuth, view_zenith,
             *, geoproject=False, ortho=False, parallel=False):
        angles = {
            'sun_azimuth': sun_azimuth, 'sun_elevation': sun_elevation,
            'view_azimuth': view_azimuth, 'view_zenith': view_zenith,
        }
        for name, value in angles.items():
            if value is None or not np.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
            angles[name] = float(value)
        if not 0 <= angles['sun_azimuth'] <= 360:
            raise ValueError('sun_azimuth must be in [0, 360] degrees')
        if not 0 <= angles['view_azimuth'] <= 360:
            raise ValueError('view_azimuth must be in [0, 360] degrees')
        if not -90 <= angles['sun_elevation'] <= 90:
            raise ValueError('sun_elevation must be in [-90, 90] degrees')
        if not 0 <= angles['view_zenith'] < 90:
            raise ValueError('view_zenith must be in [0, 90) degrees')
        if geoproject and ortho:
            raise ValueError('Choose either geographic reprojection or GLT ortho mapping')

        with h5netcdf.File(self.path, 'r') as source:
            radiance_var = source.variables['radiance']
            fill = float(radiance_var.attrs.get('_FillValue', -9999.0))
            wavelengths = np.asarray(source.groups['sensor_band_parameters'].variables['wavelengths'][:], dtype=np.float64)
            fwhm = np.asarray(source.groups['sensor_band_parameters'].variables['fwhm'][:], dtype=np.float64)
            if wavelengths.size != radiance_var.shape[2] or fwhm.size != wavelengths.size:
                raise ValueError('EMIT radiance and spectral metadata band counts differ')
            if np.any(~np.isfinite(wavelengths)) or np.any(~np.isfinite(fwhm)):
                raise ValueError('EMIT wavelength/FWHM metadata contains non-finite values')
            if np.any(np.diff(wavelengths) <= 0):
                raise ValueError('EMIT wavelengths must be strictly increasing')
            units = radiance_var.attrs.get('units', '')
            if isinstance(units, bytes):
                units = units.decode('ascii', 'replace')
            if str(units).lower().replace(' ', '') not in {'uw/cm^2/sr/nm', 'uwcm-2sr-1nm-1'}:
                raise ValueError(f'Unsupported EMIT radiance units {units!r}')
            # 1 uW/cm^2 = 10 mW/m^2. Read one spectral plane at a time so
            # processing does not hold a second full 3 GB cube in memory.
            lat = np.asarray(source.groups['location'].variables['lat'][:], dtype=np.float64)
            lon = np.asarray(source.groups['location'].variables['lon'][:], dtype=np.float64)
            elev = np.asarray(source.groups['location'].variables['elev'][:], dtype=np.float32)
            elev[elev == -9999] = np.nan
            attrs = {k: v for k, v in source.attrs.items()}
            raw_time = attrs.get('time_coverage_start')
            glt_x = glt_y = None
            if ortho:
                location = source.groups['location']
                glt_x = np.asarray(location.variables['glt_x'][:], dtype=np.int32)
                glt_y = np.asarray(location.variables['glt_y'][:], dtype=np.int32)
            shape = (glt_y.shape if ortho else lat.shape)
            cube = np.full((*shape, wavelengths.size), np.nan, dtype=np.float32)
            for band in range(wavelengths.size):
                plane = np.asarray(radiance_var[:, :, band], dtype=np.float32)
                plane[plane == fill] = np.nan
                plane *= 10.0
                if ortho:
                    valid = (glt_x > 0) & (glt_y > 0)
                    out = np.full(glt_x.shape, np.nan, dtype=np.float32)
                    # GLT values are one-based source sample and line indices.
                    out[valid] = plane[glt_y[valid] - 1, glt_x[valid] - 1]
                    cube[:, :, band] = out
                else:
                    cube[:, :, band] = plane

        if raw_time is None:
            raise ValueError('EMIT product has no time_coverage_start')
        if isinstance(raw_time, bytes):
            raw_time = raw_time.decode('ascii')
        date = datetime.fromisoformat(str(raw_time).replace('Z', '+00:00'))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        date = date.astimezone(timezone.utc)
        # NetCDF/xarray scalar time coordinates cannot serialize a Python
        # timezone-aware datetime. Store the UTC instant as naive datetime64;
        # the original UTC offset remains explicit in acquisition_date.
        time_coord = np.datetime64(date.replace(tzinfo=None), 'us')
        sensor = SensorDescription(name='EMIT', sensor_mod=Gaussian(wavelengths, fwhm))
        solar = SolarIrradiance().tsis * Misc.earth_sun_correction(date.timetuple().tm_yday)
        f0 = sensor.convolve(solar, solar_irradiance=True).rename({'wl_sensor': 'wl'})
        sza_value = 90.0 - angles['sun_elevation']
        vza_value = angles['view_zenith']
        raa_value = (angles['sun_azimuth'] - angles['view_azimuth']) % 360.0
        if ortho:
            # Geolocation follows the GLT exactly so the output pixel maps to
            # the same nearest source sample as each spectral band.
            ysafe = np.clip(glt_y - 1, 0, lat.shape[0] - 1)
            xsafe = np.clip(glt_x - 1, 0, lat.shape[1] - 1)
            lon_out = lon[ysafe, xsafe].copy()
            lat_out = lat[ysafe, xsafe].copy()
            valid = (glt_x > 0) & (glt_y > 0)
            lon_out[~valid] = np.nan
            lat_out[~valid] = np.nan
            elev_out = elev[ysafe, xsafe].copy()
            elev_out[~valid] = np.nan
        else:
            lon_out, lat_out = lon.copy(), lat.copy()
            elev_out = elev
        invalid_geo = (
            ~np.isfinite(lon_out) | ~np.isfinite(lat_out)
            | (lon_out == -9999) | (lat_out == -9999)
        )
        lon_out[invalid_geo] = np.nan
        lat_out[invalid_geo] = np.nan
        elev_out[invalid_geo] = np.nan
        cube[invalid_geo, :] = np.nan
        rtoa = np.pi * cube / (np.asarray(f0.values)[None, None, :] * np.cos(np.radians(sza_value)))
        ny, nx = rtoa.shape[:2]
        data = xr.Dataset(
            data_vars={
                'Rtoa': (('y', 'x', 'wl'), rtoa),
                'F0': (('wl',), f0.values), 'fwhm': (('wl',), fwhm),
                'lon': (('y', 'x'), lon_out), 'lat': (('y', 'x'), lat_out),
                'elevation': (('y', 'x'), elev_out),
                'sza': (('y', 'x'), np.full((ny, nx), sza_value, dtype=np.float32)),
                'saa': (('y', 'x'), np.full((ny, nx), angles['sun_azimuth'], dtype=np.float32)),
                'vza': (('y', 'x'), np.full((ny, nx), vza_value, dtype=np.float32)),
                'vaa': (('y', 'x'), np.full((ny, nx), angles['view_azimuth'], dtype=np.float32)),
                'raa': (('y', 'x'), np.full((ny, nx), raa_value, dtype=np.float32)),
            },
            coords={'y': np.arange(ny), 'x': np.arange(nx), 'wl': wavelengths, 'time': time_coord},
            attrs={
                'description': 'EMIT L1B calibrated radiance scene', 'platform': 'EMIT',
                'L1C_product_name': self.path.name, 'acquisition_date': date.isoformat(),
                'radiance_source_units': str(units), 'radiance_conversion': 'uW/cm^2/sr/nm to mW/m^2/sr/nm (x10)',
                'geometry_source': 'scene-constant angles supplied by caller',
                'ortho_source': 'EMIT GLT one-based line/sample lookup' if ortho else 'source swath',
            },
        )
        data['Rtoa'] = data.Rtoa.where(data.Rtoa >= 0)
        if geoproject:
            data.attrs['native_ground_sampling_m'] = Reproj.ground_sampling_resolution_m(lon_out, lat_out)
            data = Reproj().regridding(data, output_resolution_m=data.attrs['native_ground_sampling_m'], parallel=parallel)
        data = data.transpose('wl', 'y', 'x', missing_dims='ignore')
        for name, unit in [('F0', 'mW/m2/nm'), ('fwhm', 'nm'), ('Rtoa', '-')]:
            data[name].attrs['unit'] = unit
        self.sensor_description = sensor
        self.scene_metadata = SceneMetadata(
            platform='EMIT', acquisition_time=date, solar_zenith=data.sza,
            viewing_zenith=data.vza, relative_azimuth=data.raa,
            source_georeferencing={
                'lon_bounds': (float(np.nanmin(lon_out)), float(np.nanmax(lon_out))),
                'lat_bounds': (float(np.nanmin(lat_out)), float(np.nanmax(lat_out))),
                'crs': 'EPSG:4326',
            },
            raw={'source_paths': {'l1b_rad': str(self.path)}, 'ortho': ortho, 'geometry': angles},
        )
        return data
