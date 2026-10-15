# Hyperion L1R reader and preprocessing used by Process.execute(sensor="hyperion").
from datetime import datetime
import logging
from pathlib import Path
import re

import numpy as np
import xarray as xr

from netCDF4 import Dataset as _Dataset
from scipy.ndimage import binary_dilation as _binary_dilation, median_filter as _median_filter
from . import Misc as _Misc, Reproj as _Reproj, SolarIrradiance as _SolarIrradiance
from .config import SceneMetadata as _SceneMetadata, SensorDescription as _SensorDescription
from .spectral_sensitivity import Gaussian as _Gaussian

def _correct_local_vertical_stripes(cube, window_width, valid_mask=None, interface_mask=None,
                                    diagnostic_band_index=None, *, inplace=False):
    corrected = cube if inplace else cube.copy()
    replacement_count = 0
    interface_replacement_count = 0
    replacement_map = np.zeros(cube.shape[:2], dtype=np.uint16)
    interface_replacement_map = np.zeros(cube.shape[:2], dtype=np.uint16)
    diagnostic_replacement_map = np.zeros(cube.shape[:2], dtype=bool)
    if valid_mask is None:
        valid_mask = np.ones(cube.shape[:2], dtype=bool)
    if np.isscalar(window_width):
        widths = np.full(cube.shape[2], int(window_width), dtype=np.int32)
    else:
        widths = np.asarray(window_width, dtype=np.int32)
    if widths.shape != (cube.shape[2],):
        raise ValueError('Local destriping needs one window width or one width per band')
    for band in range(cube.shape[2]):
        width = max(1, int(widths[band]))
        image = cube[:, :, band]
        statistic_valid = np.isfinite(image) & valid_mask
        counts = statistic_valid.sum(axis=0)
        column_median = np.full(image.shape[1], np.nan, dtype=np.float64)
        # Use the actual median over valid (typically water) pixels in each
        # detector column. A local median of neighboring column medians is the
        # reference for detecting persistent local column outliers.
        for col in np.flatnonzero(counts > 0):
            column_median[col] = np.median(image[statistic_valid[:, col], col])
        local_reference = _median_filter(column_median, size=width, mode='nearest')
        local_deviation = np.abs(column_median - local_reference)
        local_mad = _median_filter(local_deviation, size=width, mode='nearest')
        threshold = 3.0 * 1.4826 * local_mad
        finite_reference = np.isfinite(local_reference) & np.isfinite(column_median)
        threshold = np.maximum(threshold, np.finfo(np.float64).eps *
                               np.maximum(1.0, np.abs(column_median)))
        stripe_columns = finite_reference & (local_deviation > threshold)
        correction = np.zeros(image.shape[1], dtype=np.float64)
        correction[stripe_columns] = (local_reference[stripe_columns]
                                      - column_median[stripe_columns])
        replace = np.isfinite(image) & stripe_columns[None, :]
        # The non-land mask controls only each column's reference statistic.
        # Once a stripe column is detected, shift every finite sample in it,
        # including high-radiometry land samples.
        corrected[:, :, band] = image + correction[None, :]
        if band == diagnostic_band_index:
            diagnostic_replacement_map = replace.copy()
        replacement_count += int(replace.sum())
        replacement_map += replace.astype(np.uint16)
        if interface_mask is not None:
            interface_replace = replace & interface_mask
            interface_replacement_count += int(interface_replace.sum())
            interface_replacement_map += interface_replace.astype(np.uint16)
    return (corrected, replacement_count, interface_replacement_count,
            replacement_map, interface_replacement_map, diagnostic_replacement_map)


def _destripe_hyperion_detectors(usable, positions, valid_mask, interface_mask,
                                 vnir_window, swir_window, *, inplace):
    """Apply local destriping to VNIR and SWIR detector sections."""
    destriped = usable if inplace else usable.copy()
    replacement_count = 0
    interface_replacement_count = 0
    replacement_map = np.zeros(usable.shape[:2], dtype=np.uint16)
    interface_replacement_map = np.zeros(usable.shape[:2], dtype=np.uint16)
    band34_replacement_map = np.zeros(usable.shape[:2], dtype=bool)
    split = np.count_nonzero(positions < 70)
    sections = (('VNIR', slice(0, split), vnir_window),
                ('SWIR', slice(split, usable.shape[2]), swir_window))
    for name, section, window_width in sections:
        if window_width < 1 or window_width % 2 == 0:
            raise ValueError('Local destriping window widths must be positive odd numbers')
        # VNIR uses narrow windows for pixel-scale stripes; SWIR uses wider
        # windows for readout-block striping.
        diagnostic_band = 34 - 8 if name == 'VNIR' else None
        (local_done, replaced, interface_replaced, replaced_map,
         interface_replaced_map, diagnostic_replaced_map) = _correct_local_vertical_stripes(
            usable[:, :, section],
            window_width,
            valid_mask,
            interface_mask,
            diagnostic_band,
            inplace=inplace,
        )
        replacement_count += replaced
        interface_replacement_count += interface_replaced
        replacement_map += replaced_map
        interface_replacement_map += interface_replaced_map
        destriped[:, :, section] = local_done
        if name == 'VNIR':
            band34_replacement_map = diagnostic_replaced_map
    return (destriped, replacement_count, interface_replacement_count,
            replacement_map, interface_replacement_map, band34_replacement_map)


def _high_radiometry_mask(cube, robust_threshold=8.0, required_bands=2):
    """Flag pixels that exceed an 8 robust-sigma threshold in >=2 VNIR channels."""
    representatives = np.asarray((5, 10, 15, 20, 30, 40))
    sample = cube[:, :, representatives].astype(np.float64)
    median = np.nanmedian(sample, axis=(0, 1))
    mad = np.nanmedian(np.abs(sample - median[None, None, :]), axis=(0, 1))
    robust_sigma = np.maximum(1.4826 * mad, np.finfo(np.float64).eps)
    high = sample > median[None, None, :] + robust_threshold * robust_sigma[None, None, :]
    valid = np.isfinite(sample)
    return (high & valid).sum(axis=2) >= required_bands


def _invalid_l1r_samples(cube, fill_value=-32767, zero_column_fraction=0.99):
    """Identify fill samples and isolated all-zero detector columns per band."""
    invalid = cube == fill_value
    zero_fraction = np.mean(cube == 0, axis=0)
    active_columns = np.any((cube != 0) & (cube != fill_value), axis=0).sum(axis=0)
    isolated_zero_column = (zero_fraction >= zero_column_fraction) & (active_columns[None, :] > 0)
    invalid |= (cube == 0) & isolated_zero_column[None, :, :]
    return invalid


def _validate_scene_angles(sun_azimuth, sun_elevation, satellite_inclination, look_angle):
    """Validate required Hyperion scene-angle inputs and return plain floats."""
    angles = {
        'sun_azimuth': sun_azimuth,
        'sun_elevation': sun_elevation,
        'satellite_inclination': satellite_inclination,
        'look_angle': look_angle,
    }
    for name, value in angles.items():
        if not np.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
        angles[name] = float(value)
    if not 0 <= angles['sun_azimuth'] <= 360:
        raise ValueError('sun_azimuth must be between 0 and 360 degrees')
    if not -90 <= angles['sun_elevation'] <= 90:
        raise ValueError('sun_elevation must be between -90 and 90 degrees')
    if not 0 < angles['satellite_inclination'] < 180:
        raise ValueError('satellite_inclination must be between 0 and 180 degrees')
    if abs(angles['look_angle']) >= 90:
        raise ValueError('look_angle magnitude must be below 90 degrees')
    return angles


def _stitch_hyperion_spectra(radiance, band_numbers, wavelengths, fwhm):
    """Keep VNIR bands up to the SWIR handover, then the calibrated SWIR bands.

    Hyperion has overlapping VNIR and SWIR channels. The old 900 nm cutoff
    discarded VNIR B55 at 905.05 nm even though the first retained SWIR band
    is B77 at 912.45 nm. Use the SWIR handover wavelength as the VNIR cutoff,
    then sort the retained channels by center wavelength.
    """
    source_band_positions = np.asarray(band_numbers) - 1
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    fwhm = np.asarray(fwhm, dtype=np.float64)
    swir = source_band_positions >= 76
    vnir = source_band_positions < 70
    if not swir.any() or not vnir.any():
        raise ValueError('Hyperion spectral metadata must include calibrated VNIR and SWIR bands')
    swir_handover = float(np.nanmin(wavelengths[swir]))
    keep = swir | (vnir & (wavelengths < swir_handover))
    selected_wavelengths = wavelengths[keep]
    order = np.argsort(selected_wavelengths, kind='stable')
    selected_wavelengths = selected_wavelengths[order]
    if not np.all(np.isfinite(selected_wavelengths)) or np.any(np.diff(selected_wavelengths) <= 0):
        raise ValueError('Stitched Hyperion wavelengths must be finite and strictly increasing')
    selected_indices = np.flatnonzero(keep)[order]
    return (
        np.take(radiance, selected_indices, axis=2),
        selected_wavelengths,
        fwhm[selected_indices],
        np.asarray(band_numbers)[selected_indices],
    )


def _hyperion_geometry(lat, sun_azimuth, sun_elevation, satellite_inclination, look_angle):
    """Estimate scene-constant solar and view angles from CLI inputs and footprint."""
    center_lat = float(np.nanmean(lat))
    inclination = np.radians(satellite_inclination)
    latitude = np.radians(center_lat)
    course_sine = np.cos(inclination) / np.cos(latitude)
    if abs(course_sine) > 1:
        raise ValueError('Satellite inclination is inconsistent with scene latitude')
    # The MET footprint establishes along-track direction (north-to-south
    # rows are descending). Inclination gives the cross-meridian heading.
    north_rows = np.nanmean(lat[0]) > np.nanmean(lat[-1])
    offset = np.degrees(np.arcsin(abs(course_sine)))
    track_azimuth = 180.0 + offset if north_rows else 360.0 - offset
    view_azimuth = (track_azimuth + (90.0 if look_angle >= 0 else -90.0)) % 360.0
    return {
        'solar_zenith': 90.0 - sun_elevation,
        'view_zenith': abs(look_angle),
        'view_azimuth': view_azimuth,
        'relative_azimuth': (sun_azimuth - view_azimuth) % 360.0,
    }



class HyperionDriver:
    """Read and prepare EO-1 Hyperion L1R data for atmospheric correction."""

    INVALID_DN = -32767
    UNCALIBRATED_BAND_RANGES = ((1, 7), (58, 76), (225, 242))

    def __init__(self, scene_path):
        scene_path = Path(scene_path)
        l1r_candidates = ([scene_path] if scene_path.is_file() and scene_path.suffix == '.L1R'
                           else sorted(scene_path.rglob('*.L1R')) if scene_path.is_dir() else [])
        if len(l1r_candidates) > 1:
            raise ValueError(f'Expected one .L1R cube under {scene_path}, found {len(l1r_candidates)}')
        if len(l1r_candidates) != 1:
            raise FileNotFoundError(f'Expected one Hyperion .L1R file in {scene_path}')
        self.l1r_path = l1r_candidates[0]

    def retrieve(self):
        """Read the L1R cube into an xarray Dataset with spectral metadata."""
        with _Dataset(str(self.l1r_path), mode='r') as source:
            variables = [value for value in source.variables.values() if value.ndim == 3]
            if len(variables) != 1:
                raise ValueError(
                    f'Expected one 3D Hyperion L1R radiance variable, found {len(variables)}'
                )
            variable = variables[0]
            variable.set_auto_maskandscale(False)
            line_band_sample = np.asarray(variable[:])
            if line_band_sample.shape[1] != 242:
                raise ValueError(f'Expected 242 spectral bands, found {line_band_sample.shape[1]}')
            if line_band_sample.shape[2] != 256:
                raise ValueError(
                    f'Expected 256 cross-track samples, found {line_band_sample.shape[2]}'
                )
            # HDF4 dimensions are along-track line, spectral band, sample.
            raw_bands = np.moveaxis(line_band_sample, 1, 0)
            metadata = {}
            for name in source.ncattrs():
                value = getattr(source, name)
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (str, int, float, np.number)):
                    metadata[name] = value
                else:
                    metadata[name] = str(value)
            header_path = self.l1r_path.with_suffix('.hdr')
            if header_path.is_file():
                text = header_path.read_text(errors='replace')
                for key in ('wavelength', 'fwhm'):
                    match = re.search(
                        rf'(?im)^\s*{key}\s*=\s*\{{(.*?)\}}', text, re.S
                    )
                    if match:
                        values = np.fromstring(match.group(1).replace(',', ' '), sep=' ')
                        if values.size:
                            metadata[key] = values.tolist()
        lines, _, samples = line_band_sample.shape
        wavelengths = metadata.pop('wavelength', None)
        fwhm = metadata.pop('fwhm', None)
        coords = {
            'band': np.arange(1, 243, dtype=np.int16),
            'y': np.arange(lines),
            'x': np.arange(samples),
        }
        if wavelengths is not None and len(wavelengths) == 242:
            coords['wavelength'] = ('band', np.asarray(wavelengths, dtype=np.float64))
        if fwhm is not None and len(fwhm) == 242:
            coords['fwhm'] = ('band', np.asarray(fwhm, dtype=np.float64))
        return xr.Dataset(
            data_vars={'raw_dn': (('band', 'y', 'x'), raw_bands)},
            coords=coords,
            attrs={
                **metadata,
                'product_level': 'L1R',
                'source_path': str(self.l1r_path),
                'source_width': samples,
                'source_height': lines,
                'source_band_count': 242,
                'source_dtype': str(raw_bands.dtype),
            },
        )

    def read(self, sun_azimuth, sun_elevation, satellite_inclination,
             look_angle, *, geoproject=True, parallel=False):
        """Read Hyperion L1R as an hGRS scene using supplied scene geometry."""
        angles = _validate_scene_angles(
            sun_azimuth, sun_elevation, satellite_inclination, look_angle
        )
        data = self.preprocess(
            sun_elevation=angles['sun_elevation'],
            include_diagnostics=False,
        )
        lines, samples = data.sizes['y'], data.sizes['x']
        lon, lat = self._met_geolocation(lines, samples)
        geometry = _hyperion_geometry(
            lat,
            angles['sun_azimuth'],
            angles['sun_elevation'],
            angles['satellite_inclination'],
            angles['look_angle'],
        )
        data['lon'] = (('y', 'x'), lon)
        data['lat'] = (('y', 'x'), lat)
        data['sza'] = (('y', 'x'), np.full((lines, samples), geometry['solar_zenith']))
        data['saa'] = (('y', 'x'), np.full((lines, samples), angles['sun_azimuth']))
        data['vza'] = (('y', 'x'), np.full((lines, samples), geometry['view_zenith']))
        data['vaa'] = (('y', 'x'), np.full((lines, samples), geometry['view_azimuth']))
        data['raa'] = (('y', 'x'), np.full((lines, samples), geometry['relative_azimuth']))
        data.attrs.update(
            geometry_source='CLI scene geometry and MET footprint',
            satellite_inclination_degrees=angles['satellite_inclination'],
            look_angle_degrees=angles['look_angle'],
        )

        if geoproject:
            native_resolution_m = _Reproj.ground_sampling_resolution_m(lon, lat)
            data.attrs['native_ground_sampling_m'] = native_resolution_m
            lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
            lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
            meters_per_degree_lat = 111_320.0
            meters_per_degree_lon = meters_per_degree_lat * np.cos(
                np.radians(0.5 * (lat_min + lat_max))
            )
            output_width = max(
                2, int(np.ceil((lon_max - lon_min) * meters_per_degree_lon
                               / native_resolution_m)) + 1
            )
            output_height = max(
                2, int(np.ceil((lat_max - lat_min) * meters_per_degree_lat
                               / native_resolution_m)) + 1
            )
            logging.info(
                'Hyperion geoprojection raster shapes: input=%s (y, x, wl), '
                'output=(%d, %d, %d) (y, x, wl); output grid=(%d, %d) (x, y)',
                (lines, samples, data.sizes['wl']), output_height, output_width,
                data.sizes['wl'], output_width, output_height,
            )
            data = _Reproj().regridding(
                data,
                output_resolution_m=native_resolution_m,
                parallel=parallel,
            )
        
        data = data.transpose('wl', 'y', 'x', missing_dims='ignore')

        data.F0.attrs.update(
            unit='mW/m2/nm',
            definition='TSIS solar irradiance adjusted for Earth-Sun distance',
        )
        data.fwhm.attrs.update(unit='nm', definition='Hyperion band full width at half maximum')
        data.Rtoa.attrs.update(unit='-', definition='Top-of-atmosphere reflectance')

        georeferencing = {
            'lon_bounds': (float(np.nanmin(lon)), float(np.nanmax(lon))),
            'lat_bounds': (float(np.nanmin(lat)), float(np.nanmax(lat))),
            'geolocation_method': 'bilinear interpolation of MET corner coordinates',
            'view_geometry_method': 'scene-constant look-angle/inclination estimate',
        }
        acquisition_time = datetime.fromisoformat(data.attrs['acquisition_date'])
        self.scene_metadata = _SceneMetadata(
            platform='Hyperion',
            acquisition_time=acquisition_time,
            solar_zenith=data['sza'],
            viewing_zenith=data['vza'],
            relative_azimuth=data['raa'],
            source_georeferencing=georeferencing,
            raw={
                'source_paths': {'l1r': str(self.l1r_path)},
                'source_geometry': angles,
            },
        )
        return data

    def _met_geolocation(self, lines, samples):
        """Interpolate pixel-center lon/lat from the four .MET corners."""
        met_path = self.l1r_path.with_suffix('.MET')
        if not met_path.is_file():
            raise FileNotFoundError(f'Hyperion geolocation metadata not found: {met_path}')
        text = met_path.read_text(errors='replace')
        corners = {}
        for key in ('UL', 'UR', 'LL', 'LR'):
            for coord in ('LAT', 'LON'):
                match = re.search(
                    rf'PRODUCT_{key}_CORNER_{coord}\s*=\s*([-+0-9.eE]+)', text
                )
                if match is None:
                    raise ValueError(f'Missing PRODUCT_{key}_CORNER_{coord} in {met_path}')
                corners[(key, coord)] = float(match.group(1))
        row = ((np.arange(lines, dtype=np.float64) + 0.5) / lines)[:, None]
        col = ((np.arange(samples, dtype=np.float64) + 0.5) / samples)[None, :]

        def interpolate(coord):
            upper = ((1.0 - col) * corners[('UL', coord)]
                     + col * corners[('UR', coord)])
            lower = ((1.0 - col) * corners[('LL', coord)]
                     + col * corners[('LR', coord)])
            return (1.0 - row) * upper + row * lower

        return interpolate('LON'), interpolate('LAT')

    def preprocess(self, apply_land_mask=True, vnir_local_window=5,
                   swir_local_window=41, *, include_diagnostics=False,
                   sun_elevation=None):
        """Prepare Hyperion L1R data as an xarray Dataset.

        Atmospheric processing passes ``sun_elevation`` and receives sensor-
        stitched ``Rtoa``. Diagnostic mode retains the intermediate
        destriped radiance for the test visualization workflow.
        """
        if not include_diagnostics and sun_elevation is None:
            raise ValueError('sun_elevation is required when diagnostic output is disabled')
        if sun_elevation is not None:
            sun_elevation = float(sun_elevation)
            if not np.isfinite(sun_elevation) or not -90 <= sun_elevation <= 90:
                raise ValueError('sun_elevation must be finite and between -90 and 90 degrees')
        source = self.retrieve()
        raw_cube = source['raw_dn'].transpose('y', 'x', 'band').values
        # Keep the source cross-track order and all 256 samples. The netCDF4
        # reader exposes the stored line/band/sample dimensions directly; no
        # circular shift or edge-column removal is applied.
        ordered = raw_cube
        uncalibrated = np.concatenate([
            np.arange(first - 1, last)
            for first, last in self.UNCALIBRATED_BAND_RANGES
        ])
        positions = np.setdiff1d(np.arange(242), uncalibrated)
        if include_diagnostics:
            radiance = ordered.astype(np.float64)
            invalid_samples = _invalid_l1r_samples(ordered, self.INVALID_DN)
            radiance[invalid_samples] = np.nan
            radiance[:, :, 7:57] /= 40.0
            radiance[:, :, 76:224] /= 80.0
            usable = radiance[:, :, positions].copy()
        else:
            # The atmospheric path only needs calibrated bands. Avoid making
            # a float64 copy of all 242 source bands and a second copy of the
            # retained bands.
            usable = np.take(ordered, positions, axis=2).astype(np.float64)
            invalid_samples = _invalid_l1r_samples(usable, self.INVALID_DN)
            usable[invalid_samples] = np.nan
            vnir = (positions >= 7) & (positions < 57)
            swir = (positions >= 76) & (positions < 224)
            usable[:, :, vnir] /= 40.0
            usable[:, :, swir] /= 80.0
            # Only compact metadata/profile/path are needed beyond this point.
            del source['raw_dn'], raw_cube, ordered
        land_mask = _high_radiometry_mask(usable)
        valid_mask = ~land_mask if apply_land_mask else np.ones(land_mask.shape, dtype=bool)
        land_water_interface = _binary_dilation(
            land_mask, structure=np.ones((3, 3)), iterations=1
        ) & ~land_mask
        metadata = source.attrs

        def spectral_metadata(name):
            if name not in source.coords or source[name].size != 242:
                return None
            return np.asarray(source[name].values)[positions]

        wavelengths = spectral_metadata('wavelength')
        fwhm = spectral_metadata('fwhm')
        if not include_diagnostics and (wavelengths is None or fwhm is None):
            raise ValueError(
                'Hyperion spectral sidecar must contain 242 wavelength and fwhm values'
            )
        (destriped, replacement_count, interface_replacement_count,
         replacement_map, interface_replacement_map,
         band34_local_replacements) = _destripe_hyperion_detectors(
            usable,
            positions,
            valid_mask if apply_land_mask else None,
            land_water_interface,
            vnir_local_window,
            swir_local_window,
            inplace=not include_diagnostics,
        )
        attrs = {
            **metadata,
            'local_destriping_vnir_window': int(vnir_local_window),
            'local_destriping_swir_window': int(swir_local_window),
            'land_mask_applied': bool(apply_land_mask),
        }
        coords = {
            'y': source.y.values,
            'x': source.x.values,
            'band': positions + 1,
            'source_band_index': ('band', positions),
        }
        wavelengths = spectral_metadata('wavelength')
        fwhm_values = spectral_metadata('fwhm')
        if wavelengths is not None:
            coords['wavelength'] = ('band', wavelengths)
        if fwhm_values is not None:
            coords['fwhm'] = ('band', fwhm_values)
        if include_diagnostics:
            data_vars = {
                'raw_dn': (('y', 'x', 'source_band'), raw_cube),
                'destriped': (('y', 'x', 'band'), destriped),
                'radiance': (('y', 'x', 'band'), usable),
                'land_mask': (('y', 'x'), land_mask),
                'invalid_data_mask': (
                    ('y', 'x'), np.any(invalid_samples[:, :, positions], axis=2)
                ),
                'valid_mask': (('y', 'x'), valid_mask),
                'land_water_interface': (('y', 'x'), land_water_interface),
                'local_replacement_map': (('y', 'x'), replacement_map),
                'band34_local_replacement_map': (
                    ('y', 'x'), band34_local_replacements
                ),
                'local_interface_replacement_map': (
                    ('y', 'x'), interface_replacement_map
                ),
            }
            coords['source_band'] = source.band.values
            attrs.update({
                'local_replacement_count': replacement_count,
                'local_interface_replacement_count': interface_replacement_count,
                'source_shape': tuple(raw_cube.shape),
                'ordered_shape': tuple(ordered.shape),
            })
        else:
            # The atmospheric Dataset contains only the sensor-ready radiance,
            # reflectance, and spectral coordinates needed by Process.
            (
                radiance_cube,
                wavelengths,
                fwhm,
                stitched_band_numbers,
            ) = _stitch_hyperion_spectra(destriped, positions + 1, wavelengths, fwhm)
            acquisition_time = datetime.strptime(
                str(metadata['ImageStartTime']), '%Y%j%H%M%S.%f'
            )
            self.sensor_description = _SensorDescription(
                name='Hyperion',
                sensor_mod=_Gaussian(wavelengths, fwhm),
            )
            solar_irradiance = (
                _SolarIrradiance().tsis
                * _Misc.earth_sun_correction(acquisition_time.timetuple().tm_yday)
            )
            f0 = self.sensor_description.convolve(
                solar_irradiance, solar_irradiance=True
            ).rename({'wl_sensor': 'wl'})
            reflectance_cube = np.pi * radiance_cube / (
                f0.values[None, None, :]
                * np.cos(np.radians(90.0 - sun_elevation))
            )
            data_vars = {
                'Rtoa': (('y', 'x', 'wl'), reflectance_cube),
                'F0': (('wl',), f0.values),
                'fwhm': (('wl',), fwhm),
            }
            coords = {
                'y': source.y.values,
                'x': source.x.values,
                'wl': wavelengths,
                'source_band_number': ('wl', stitched_band_numbers),
                'time': acquisition_time,
            }
            attrs.update({
                'description': 'EO-1 Hyperion L1R radiance scene',
                'platform': 'Hyperion',
                'L1C_product_name': self.l1r_path.name,
                'acquisition_date': acquisition_time.isoformat(),
                'radiance_preprocessing': 'calibrated bands, local destriping',
            })
            result = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            source.close()
            del source, usable, invalid_samples, destriped
            return result
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
