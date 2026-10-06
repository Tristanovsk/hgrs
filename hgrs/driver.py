import os
import glob

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

import h5py
import xml.etree.ElementTree as ET

from scipy.interpolate import RegularGridInterpolator

import datetime as dt
import logging

from . import SolarIrradiance, Reproj, Misc
from .spectral_sensitivity import BaselineInterp, Gaussian
from .config import SceneMetadata, SensorDescription
from .hyperion import HyperionDriver
from .emit import EmitDriver

class Driver():
    def __init__(self,
                 satellite='enmap'):

        self.satellite = satellite
        self.scene_metadata = None

        if 'prisma' in satellite:
            self.driver = self.read_prisma
        elif 'enmap' in satellite:
            self.driver = self.read_l1c_enmap
        elif 'tanager' in satellite:
            self.driver = self.read_tanager
        elif 'emit' in satellite:
            self.driver = self.read_emit
        else:
            logging.info('satellite mission not recognized, stop')
            return



    def read_prisma(self,
                    l1c_path: str,
                    l2c_path: str,
                    reflectance_unit=True,
                    drop_vars=True,
                    geoproject=True,
                    parallel=False
                    ):
        """Read a PRISMA scene in the layout expected by ``Process``.

        The returned dataset must provide spectral data as ``(wl, y, x)``;
        ``sza``, ``vza``, ``raa``, and the mask layers use ``(y, x)``. ``F0``
        and ``fwhm`` use ``(wl,)``. With ``geoproject=False``, keep the native
        pixel ``x``/``y`` coordinates and the 2D ``lon``/``lat`` geolocation
        arrays so the output can be projected later. With geoprojection,
        ``Reproj`` supplies the target-grid ``x``/``y`` coordinates instead.
        """
        logging.info('construct L1C image plus angle rasters')
        try:
            dc_l1c = self.read_l1c_prisma(l1c_path,
                                          reflectance_unit=reflectance_unit,
                                          drop_vars=drop_vars)
            dc_l2c = self.read_l2c_prisma(l2c_path)
        except Exception as e:
            logging.info(f'input file format not recognized {l1c_path}, {l2c_path}, stop')
            raise RuntimeError from e

        for param in ['sza', 'vza', 'raa']:
            dc_l1c[param] = dc_l2c[param]
        del dc_l2c

        source_georeferencing = self._source_georeferencing(dc_l1c)

        # dc_l1c = dc_l1c.chunk({'x': 200, 'y': 200, 'wl': 10})

        if geoproject:
            dc_l1c = Reproj().regridding(dc_l1c, parallel=parallel)
        else:
            # Enforce the spectral dimension order in the output contract;
            # spatial-only variables remain on (y, x).
            dc_l1c = dc_l1c.transpose("wl", "y", "x", missing_dims="ignore")

        self.scene_metadata = self._make_scene_metadata(
            dc_l1c,
            platform='PRISMA',
            source_georeferencing=source_georeferencing,
            source_paths={'l1c': l1c_path, 'l2c': l2c_path},
        )

        return dc_l1c

    @staticmethod
    def _source_georeferencing(dataset):
        """Capture compact source-grid information before reprojection."""
        source = {}
        for name in ('lon', 'lat'):
            if name in dataset:
                coordinate = dataset[name]
                source[f'{name}_bounds'] = (
                    float(coordinate.min().values),
                    float(coordinate.max().values),
                )
        try:
            crs = dataset.rio.crs
        except (AttributeError, RuntimeError):
            crs = None
        if crs is not None:
            source['crs'] = str(crs)
        for name in ('x', 'y'):
            if name in dataset.coords and dataset[name].size:
                source[f'{name}_bounds'] = (
                    float(dataset[name].min().values),
                    float(dataset[name].max().values),
                )
        return source

    @staticmethod
    def _make_scene_metadata(
        dataset,
        platform,
        source_georeferencing=None,
        source_paths=None,
    ):
        time = dataset['time']
        source_georeferencing = dict(source_georeferencing or {})
        if not source_georeferencing:
            raise ValueError(f'{platform} source georeferencing was not captured')
        raw = {
            'product_name': dataset.attrs.get('L1C_product_name'),
            'source_paths': dict(source_paths or {}),
        }
        return SceneMetadata(
            platform=platform,
            acquisition_time=time,
            solar_zenith=dataset['sza'],
            viewing_zenith=dataset['vza'],
            relative_azimuth=dataset['raa'],
            source_georeferencing=source_georeferencing,
            raw=raw,
        )
    def read_l1c_prisma(self,
                        l1c_path: str,
                        reflectance_unit=False,
                        drop_vars=False):
        '''
        Load PRISMA L1C data into xarray rasters
        :param l1c_path: absolute path to the .h5 prisma file
        :param reflectance_unit: to convert from TOA radiance to TOA reflectance
        :param drop_vars: if True remove the radiance raster to keep reflectance only
        :return:
        '''
        # =============================================================================
        # Load geolocation, solar irradiance and TOA radiance
        # =============================================================================
        ds = h5py.File(l1c_path)

        # coarse geometry
        sza = ds.attrs["Sun_zenith_angle"]

        # Geolocation
        lat = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR"][:].T
        lon = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR"][:].T
        xdim, ydim = lat.shape

        # Wavelength / fwhm
        wl = ds.attrs["List_Cw_Vnir"][5:]
        wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
        fwhm = ds.attrs["List_Fwhm_Vnir"][5:]
        fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
        sort_index = np.argsort(wl)
        wl = wl[sort_index]
        fwhm = fwhm[sort_index]
        fwhm = xr.DataArray(data=fwhm, name='fwhm',
                            coords=dict(wl=wl),
                            attrs=dict(description="PRISMA relative spectral response parameter"))
        self.sensor_description = SensorDescription(
            name='PRISMA',
            sensor_mod=Gaussian(wl, fwhm.values),

        )

        # solar irradiance convolution to the PRISMA spectral response function and scaled
        # by the day of the year
        solar_irr = SolarIrradiance()
        F0 = solar_irr.tsis  # huillier # gueymard # kurucz
        date_str = ds.attrs["Product_StartTime"].decode('UTF-8')

        DOY = dt.datetime.strptime(date_str,
                                   "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
        # get correction for Sun-Earth distance and correct solar irradiance
        D2 = Misc.earth_sun_correction(DOY)
        F0 = F0 * D2
        F0_sensor = self.sensor_description.convolve(
            F0, solar_irradiance=True
        ).rename({'wl_sensor': 'wl'})
        F0_sensor.name = 'F0'
        F0_sensor.attrs = {
            'description': 'Convolved solar irradiance from TSIS data',
            'unit': 'mW/m2/nm',
        }
        # DN to TOA radiance
        gain = {"vnir": ds.attrs["ScaleFactor_Vnir"],
                "swir": ds.attrs["ScaleFactor_Swir"]}

        # -------------------------------------------------------------------------------
        VNIR = np.moveaxis(ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"][:, 5:, :] / gain["vnir"],
                           [0, 1, 2],
                           [1, 2, 0])
        SWIR = np.moveaxis(ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][:, :-2, :] / gain["swir"],
                           [0, 1, 2],
                           [1, 2, 0])
        Ltoa = np.dstack((VNIR, SWIR))
        del VNIR, SWIR

        Ltoa = Ltoa[:, :, sort_index]

        data = xr.Dataset(data_vars=dict(Ltoa=(["y", "x", "wl"], Ltoa),
                                         F0=(['wl'], F0_sensor.values),
                                         fwhm=(['wl'], fwhm.values),
                                         lon=(["y", "x"], lon),
                                         lat=(["y", "x"], lat)),
                          coords=dict(
                              x=np.arange(xdim)[::-1],
                              y=np.arange(ydim)[::-1],
                              time=dt.datetime.strptime(date_str,
                                                        "%Y-%m-%dT%H:%M:%S.%f"),
                              wl=wl),
                          attrs=dict(description="PRISMA L1C cube data"))

        # chunk data for dask
        # data = data.chunk({'x': 200, 'y': 200, 'wl': -1})

        # TODO check errors due to bulk SZA value instead of per pixel values
        if reflectance_unit:
            data['Rtoa'] = np.pi * data.Ltoa / (data.F0 * np.cos(np.radians(sza)))
        if drop_vars:
            data = data.drop_vars('Ltoa')

        # =============================================================================
        # Load other metadata
        # =============================================================================
        data.attrs["L1C_product_name"] = os.path.basename(l1c_path)
        data.attrs["acquisition_date"] = date_str
        data.attrs["sza"] = ds.attrs["Sun_zenith_angle"]
        data.attrs["saa"] = ds.attrs["Sun_azimuth_angle"]
        data.F0.attrs['unit'] = 'mW/m2/nm'
        data.F0.attrs['definition'] = 'Solar irradiance corrected for Sun-Earth distance'

        # =============================================================================
        # Load masks
        # =============================================================================
        data = data.assign(cloud_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/Cloud_Mask"][:].T))
        data = data.assign(sunglint_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SunGlint_Mask"][:].T))
        data = data.assign(landcover_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/LandCover_Mask"][:].T))

        return data.sel(wl=slice(350, 2550))

    def read_l2c_prisma(self,
                        l2c_path: str):
        '''
        Load PRISMA L2C data (including observation angles) into xarray rasters
        :param l2c_path: absolute path to the .h5 prisma file
        :return:
        '''

        # =============================================================================
        # Load geolocation, solar irradiance and TOA radiance
        # =============================================================================
        ds = h5py.File(l2c_path)
        # Geolocation
        lat = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Latitude"][:].T
        lon = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Longitude"][:].T
        xdim, ydim = lat.shape

        # Wavelength / fwhm
        wl = ds.attrs["List_Cw_Vnir"][3:]
        wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
        fwhm = ds.attrs["List_Fwhm_Vnir"][3:]
        fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
        sort_index = np.argsort(wl)
        wl = wl[sort_index]
        fwhm = fwhm[sort_index]

        # # Thuillier solar irradiance convolved to the PRISMA ISRF and scaled
        # # by the day of the year
        # I0 = load_thuillier_solar_spectrum(wl, fwhm)
        # DOY = dt.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),
        #                                  "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
        # U = 1 - 0.01672 * np.cos(0.9856 * (DOY - 4))
        # I0 = I0 * U

        # DN to TOA radiance
        gain = {"vnir_min": ds.attrs["L2ScaleVnirMin"],
                "vnir_max": ds.attrs["L2ScaleVnirMax"],
                "swir_min": ds.attrs["L2ScaleSwirMin"],
                "swir_max": ds.attrs["L2ScaleSwirMax"]}

        # -------------------------------------------------------------------------------
        VNIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/VNIR_Cube"][:, 3:, :]
        VNIR = gain["vnir_min"] + VNIR * (gain["vnir_max"] - gain["vnir_min"]) / 65535
        VNIR = np.moveaxis(VNIR, [0, 1, 2], [1, 2, 0])
        SWIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube"][:, :-2, :]
        SWIR = gain["swir_min"] + SWIR * (gain["swir_max"] - gain["swir_min"]) / 65535
        SWIR = np.moveaxis(SWIR, [0, 1, 2], [1, 2, 0])
        # print(f"VNIR shape = {VNIR.shape}")
        # print(f"SWIR shape = {SWIR.shape}")
        rho = np.dstack((VNIR, SWIR))
        rho = rho[:, :, sort_index]
        del VNIR, SWIR

        # # -------------------------------------------------------------------------------
        data = xr.Dataset(data_vars=dict(rho=(["y", "x", "wl"], rho), lon=(["y", "x"], lon),
                                         lat=(["y", "x"], lat)),
                          coords=dict(
                              x=np.arange(xdim)[::-1],
                              y=np.arange(ydim)[::-1],

                              wl=wl),
                          attrs=dict(description="PRISMA L2C cube data"))

        # =============================================================================
        # Load other metadata
        # =============================================================================
        data.attrs["L2C_product_name"] = os.path.basename(l2c_path)
        data.attrs["acquisition_date"] = ds.attrs["Product_StartTime"].decode('UTF-8')

        # =============================================================================
        # Load geometries
        # =============================================================================
        data = data.assign(vza=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Observing_Angle"][:].T))
        data = data.assign(raa=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Rel_Azimuth_Angle"][:].T))
        data = data.assign(sza=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Solar_Zenith_Angle"][:].T))

        # =============================================================================
        # Read atmospheric data
        # =============================================================================
        hdf_variables = ["AOT", "AEX", "WVM", "COT"]
        ds_variables = ["aot", "aex", "wvm", "cot"]
        dims = {"AOT": ["y2", "x2"],
                "AEX": ["y2", "x2"],
                "WVM": ["y", "x"],
                "COT": ["y", "x"]}
        for ii, var in enumerate(hdf_variables):
            gain_min = ds.attrs[f"L2Scale{var}Min"]
            gain_max = ds.attrs[f"L2Scale{var}Max"]
            matrix = ds[f"/HDFEOS/SWATHS/PRS_L2C_{var}/Data Fields/{var}_Map"][:].T
            var_dims = dims[var]
            data = eval(f'data.assign({ds_variables[ii]}=({var_dims},gain_min + matrix*(gain_max-gain_min)/65535))')
        x2dim, y2dim = data['aot'].shape
        data = data.assign_coords({'x2': np.arange(x2dim)[::-1], 'y2': np.arange(y2dim)[::-1]})

        return data

    def read_tanager(self, l1c_path: str, reflectance_unit=False,
                     drop_vars=False, geoproject=True, parallel=False):
        """Read a Planet Tanager basic-radiance HDF5 scene.

        Return an xarray dataset using the hGRS input convention: ``Ltoa`` is
        ``(y, x, wl)``, image geometry is ``(y, x)``, and ``F0``/``fwhm`` are
        ``(wl,)``. The source HDF5 radiance is stored band-first.
        """
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

        source_georeferencing = self._source_georeferencing(data)
        if geoproject:
            native_resolution_m = Reproj.ground_sampling_resolution_m(lon, lat)
            data.attrs['native_ground_sampling_m'] = native_resolution_m
            data = Reproj().regridding(
                data,
                output_resolution_m=native_resolution_m,
                parallel=parallel,
            )
        self.scene_metadata = self._make_scene_metadata(
            data,
            platform='Tanager',
            source_georeferencing=source_georeferencing,
            source_paths={'l1c': l1c_path},
        )
        return data

    def read_emit(self, l1b_path: str, sun_azimuth, sun_elevation,
                  view_azimuth, view_zenith, *, geoproject=False,
                  ortho=False, parallel=False):
        """Read an EMIT L1B_RAD granule with explicit scene geometry."""
        reader = EmitDriver(l1b_path)
        data = reader.read(
            sun_azimuth, sun_elevation, view_azimuth, view_zenith,
            geoproject=geoproject, ortho=ortho, parallel=parallel,
        )
        self.sensor_description = reader.sensor_description
        self.scene_metadata = reader.scene_metadata
        return data

    def read_l1c_enmap(self,
                       l1c_path: str,
                       reflectance_unit=False,
                       drop_vars=False,
                       filter_bad_bands=True
                       ):

        for ext in ['BIL','TIF']:
            l1c_raster_path = glob.glob(os.path.join(l1c_path, "*SPECTRAL_IMAGE."+ext))
            if len(l1c_raster_path)>0:
                l1c_raster_path =l1c_raster_path[0]
                break

        metadata_path = glob.glob(os.path.join(l1c_path, "*METADATA*.XML"))[0]

        # get XML metadata
        tree = ET.parse(metadata_path)
        root = tree.getroot()
        specific = root.find('specific')
        fwhm, offset, gain = {}, {}, {}
        for child in root.find('specific').find('bandCharacterisation'):
            wl = child.findtext('wavelengthCenterOfBand')
            fwhm[wl] = child.findtext('FWHMOfBand')
            offset[wl] = child.findtext('OffsetOfBand')
            gain[wl] = child.findtext('GainOfBand')

        fwhm = pd.DataFrame(fwhm.items(), columns=['wl', 'fwhm']).astype(float).set_index('wl').to_xarray()
        offset = pd.DataFrame(offset.items(), columns=['wl', 'offset']).astype(float).set_index('wl').to_xarray()
        gain = pd.DataFrame(gain.items(), columns=['wl', 'gain']).astype(float).set_index('wl').to_xarray()
        metadata = xr.merge([fwhm, offset, gain])

        # get respective indexes of vnir and swir sensors
        self.vnir_idx = np.array(root.find('specific').find('vnirProductQuality'
                                            ).findtext('expectedChannelsList').split(',')).astype(int) - 1
        self.swir_idx= np.array(root.find('specific').find('swirProductQuality'
                                            ).findtext('expectedChannelsList').split(',')).astype(int) - 1

        ### set wavelength to keep (EnMAP shows strange values over the overlap between vnir and swir sensors
        self.vnir_idx_tokeep = self.vnir_idx[:-13]
        self.swir_idx_tokeep = self.swir_idx[3:]

        date_str = specific.findtext('datatakeStart').strip().replace("Z", "")

        # open raster
        data = rxr.open_rasterio(l1c_raster_path, chunks={'x': 512, 'y': 512, 'band': 20},
                                 mask_and_scale=True).to_dataset(name='Ltoa')

        if '.BIL' in l1c_raster_path:
            data = data.swap_dims({'band': 'wavelength'}).rename({'wavelength': 'wl'})
        else:
            # get FWHM data and scale radiance
            data = data.rename({'band': 'wl'})
            data['wl'] = metadata.wl
            data['fwhm'] = metadata.fwhm
            data['Ltoa'] = metadata.gain * data['Ltoa'] + metadata.offset


        data = data.transpose("wl", "y", "x")
        # data['time'] = dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        # data = data.set_coords('time')

        # convert from W.m-2.sr-1.nm-1 to mW.m-2.sr-1.nm-1
        data['Ltoa'] = 1e3 * data['Ltoa'].drop_attrs()
        data['Ltoa'].attrs['unit'] = 'mW.m-2.sr-1.nm-1'
        data['Ltoa'].attrs['description'] = 'top-of-atmosphere radiance'
        data['Ltoa'].attrs['name'] = 'radiance'

        x = data.x.values
        y = data.y.values

        x_coarse = [x[0], x[-1]]
        y_coarse = [y[0], y[-1]]
        XX, YY = np.meshgrid(x, y)
        points = np.stack([YY.ravel(), XX.ravel()], axis=-1)

        sza_values = np.zeros((2, 2))
        sza_values[0, 0] = 90 - float(specific.find('sunElevationAngle').findtext('upper_left'))
        sza_values[0, 1] = 90 - float(specific.find('sunElevationAngle').findtext('upper_right'))
        sza_values[1, 0] = 90 - float(specific.find('sunElevationAngle').findtext('lower_left'))
        sza_values[1, 1] = 90 - float(specific.find('sunElevationAngle').findtext('lower_right'))
        sza_interp = RegularGridInterpolator((y_coarse, x_coarse), sza_values, method='linear')
        sza = sza_interp(points).reshape(YY.shape)

        saa_values = np.zeros((2, 2))
        saa_values[0, 0] = float(specific.find('sunAzimuthAngle').findtext('upper_left'))
        saa_values[0, 1] = float(specific.find('sunAzimuthAngle').findtext('upper_right'))
        saa_values[1, 0] = float(specific.find('sunAzimuthAngle').findtext('lower_left'))
        saa_values[1, 1] = float(specific.find('sunAzimuthAngle').findtext('lower_right'))
        saa_interp = RegularGridInterpolator((y_coarse, x_coarse), saa_values, method='linear')
        saa = saa_interp(points).reshape(YY.shape)

        vza_values = np.zeros((2, 2))
        vza_values[0, 0] = float(specific.find('viewingZenithAngle').findtext('upper_left'))
        vza_values[0, 1] = float(specific.find('viewingZenithAngle').findtext('upper_right'))
        vza_values[1, 0] = float(specific.find('viewingZenithAngle').findtext('lower_left'))
        vza_values[1, 1] = float(specific.find('viewingZenithAngle').findtext('lower_right'))
        vza_interp = RegularGridInterpolator((y_coarse, x_coarse), vza_values, method='linear')
        vza = vza_interp(points).reshape(YY.shape)

        vaa_values = np.zeros((2, 2))
        vaa_values[0, 0] = float(specific.find('viewingAzimuthAngle').findtext('upper_left'))
        vaa_values[0, 1] = float(specific.find('viewingAzimuthAngle').findtext('upper_right'))
        vaa_values[1, 0] = float(specific.find('viewingAzimuthAngle').findtext('lower_left'))
        vaa_values[1, 1] = float(specific.find('viewingAzimuthAngle').findtext('lower_right'))
        vaa_interp = RegularGridInterpolator((y_coarse, x_coarse), vaa_values, method='linear')
        vaa = vaa_interp(points).reshape(YY.shape)

        raa = (saa - vaa) % 360

        solar_irr = SolarIrradiance()
        F0 = solar_irr.tsis  # thuillier # gueymard # kurucz

        ## Compute irradiance for the Day Of the Year (date of acquisition)
        DOY = dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
        # get correction for Sun-Earth distance and correct solar irradiance
        D2 = Misc.earth_sun_correction(DOY)
        F0 = F0 * D2
        self.F0 = F0

        # Preserve the established EnMAP two-stage irradiance resampling.
        self.sensor_description = SensorDescription(
            name='EnMAP',
            sensor_mod=Gaussian(data.wl.values, data.fwhm.values),

        )
        F0 = self.sensor_description.convolve(F0).rename({'wl_sensor': 'wl'})

        # Apply the same sensor response operator used later in atmospheric correction.
        F0_sensor = self.sensor_description.convolve(
            F0, solar_irradiance=True
        ).rename({'wl_sensor': 'wl'})
        F0_sensor.name = 'F0'
        F0_sensor.attrs = {
            'description': 'Convolved solar irradiance from TSIS data',
            'unit': 'mW/m2/nm',
        }

        d_vars = dict(
            Ltoa=data.Ltoa,
            F0=(['wl'], F0_sensor.values),
            fwhm=(['wl'], data.fwhm.values),
            sza=(["y", "x"], sza),
            saa=(["y", "x"], saa),
            vza=(["y", "x"], vza),
            vaa=(["y", "x"], vaa),
            raa=(['y', 'x'], raa),
        )

        data = xr.Dataset(data_vars=d_vars,
                          coords=dict(
                              x=data.x.values,
                              y=data.y.values,
                              wl=data.wl.values,
                              time=dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")),

                          attrs=dict(description="EnMAP L1C cube data"))

        ## Compute Top of Atmosphere Reflectance
        if reflectance_unit:
            ## WARNING we take the mean SZA value to save time/memory               ##
            ## this could  induce 0.1% uncertainty onn the Ltoa to Rtoa  conversion ##
            mu0 = np.cos(np.radians(data.sza.mean()))
            data['Rtoa'] = np.pi * data.Ltoa / (data.F0 * mu0)
            if drop_vars:
                data = data.drop_vars('Ltoa')

        # discard angles outside the image frame
        params = ['sza', 'vza', 'raa', 'saa', 'vaa']
        for i in range(len(params)):
            data[params[i]] = data[params[i]].where(data.Rtoa.isel(wl=0) > 0)

        wl_vnir = data.wl.isel(wl=self.vnir_idx).values
        wl_swir = data.wl.isel(wl=self.swir_idx).values
        if filter_bad_bands:
            wl_vnir = data.wl.isel(wl=self.vnir_idx_tokeep).values
            wl_swir = data.wl.isel(wl=self.swir_idx_tokeep).values
            data =data.isel(wl=[*self.vnir_idx_tokeep,*self.swir_idx_tokeep])

        # Attributes

        data.attrs["L1C_product_name"] = os.path.basename(l1c_path)
        data.attrs["acquisition_date"] = dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        data.attrs["vnir_index"] = self.vnir_idx
        data.attrs["swir_index"] = self.swir_idx
        data.attrs["vnir_index_tokeep"] = self.vnir_idx_tokeep
        data.attrs["swir_index_tokeep"] = self.swir_idx_tokeep
        data.attrs["vnir_bands"] = wl_vnir
        data.attrs["swir_bands"] = wl_swir

        data.sza.attrs['definition'] = " Sun Zenith Angle"
        data.saa.attrs['definition'] = " Sun Azimuth Angle"
        data.vza.attrs['definition'] = " Viewing Zenith Angle"
        data.vaa.attrs['definition'] = " Viewing Azimuth Angle"
        data.raa.attrs['definition'] = " Relative Azimuth Angle"

        data.sza.attrs['unit'] = "degree"
        data.saa.attrs['unit'] = "degree"
        data.vza.attrs['unit'] = "degree"
        data.vaa.attrs['unit'] = "degree"
        data.raa.attrs['unit'] = "degree"

        data.F0.attrs['unit'] = 'mW/m2/nm'
        data.Rtoa.attrs['definition'] = 'Reflectance at the Top of the Atmosphere'
        data.Rtoa.attrs['unit'] = '-'
        data.F0.attrs['definition'] = 'Solar irradiance corrected for Sun-Earth distance'
        data.Ltoa.attrs['unit'] = 'mW/m2/sr/nm'
        data.Ltoa.attrs['definition'] = 'Top-of-atmosphere radiance'
        data.fwhm.attrs['unit'] = 'nm'
        data.fwhm.attrs['definition'] = 'Full Width at Half Maximum'
        # data.attrs['crs'] = CRS.from_epsg(32630)

        self.scene_metadata = self._make_scene_metadata(
            data,
            platform='EnMAP',
            source_georeferencing=self._source_georeferencing(data),
            source_paths={'l1c': l1c_path, 'metadata': metadata_path},
        )

        return data
