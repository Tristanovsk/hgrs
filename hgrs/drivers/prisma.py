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


class PrismaDriver(L1Driver):
    r'''
    Reader of the PRISMA L1 products, with the viewing angles of the L2C product.

    The TOA radiance :math:`L_{TOA}` (mW m\ :sup:`-2` sr\ :sup:`-1` nm\ :sup:`-1`) is converted into TOA
    reflectance

    .. math::

       R_{TOA}(\lambda) = \frac{\pi\, L_{TOA}(\lambda)}{F_0(\lambda)\, \cos\theta_s}

    where :math:`F_0` is the TSIS-1 solar irradiance (Coddington et al., 2021) corrected for the
    Earth-Sun distance (:py:meth:`~hgrs.utils.Misc.earth_sun_correction`) and convolved with the
    spectral response of the bands, and :math:`\theta_s` the solar zenith angle of the scene.

    Examples
    --------
    .. code-block:: python

       from hgrs.drivers import PrismaDriver

       # PRISMA L1 + L2C (viewing angles), geoprojected onto a longitude-latitude grid
       l1c = PrismaDriver().read(['PRS_L1_STD_OFFL_..._0001.he5', 'PRS_L2C_STD_..._0001.he5'])
       l1c.Rtoa.sel(wl=[650, 560, 480], method='nearest').plot.imshow(rgb='wl', robust=True)

    References
    ----------
    * Coddington, O. M., Richard, E. C., Harber, D., et al. (2021). The TSIS-1 hybrid solar reference
      spectrum. *Geophysical Research Letters*, 48(12), e2020GL091709. https://doi.org/10.1029/2020GL091709
    * Cogliati, S., Sarti, F., Chiarantini, L., et al. (2021). The PRISMA imaging spectroscopy mission:
      overview and first performance analysis. *Remote Sensing of Environment*, 262, 112499.
      https://doi.org/10.1016/j.rse.2021.112499
    '''

    name = 'prisma'
    platform = 'PRISMA'

    def read(self,
             img_path,
             *,
             geoproject=True,
             reflectance_unit=True,
             drop_vars=True,
             parallel=False
             ):
        """Read a PRISMA scene in the layout expected by ``Process``.

        ``img_path`` is the pair ``[l1_path, l2c_path]`` of the L1 and L2C files.

        The returned dataset must provide spectral data as ``(wl, y, x)``;
        ``sza``, ``vza``, ``raa``, and the mask layers use ``(y, x)``. ``F0``
        and ``fwhm`` use ``(wl,)``. With ``geoproject=False``, keep the native
        pixel ``x``/``y`` coordinates and the 2D ``lon``/``lat`` geolocation
        arrays so the output can be projected later. With geoprojection,
        ``Reproj`` supplies the target-grid ``x``/``y`` coordinates instead.
        """
        l1c_path, l2c_path = img_path
        logger.info('construct L1C image plus angle rasters')
        try:
            dc_l1c = self.read_l1c(l1c_path,
                                   reflectance_unit=reflectance_unit,
                                   drop_vars=drop_vars)
            dc_l2c = self.read_l2c(l2c_path)
        except Exception as e:
            logger.info(f'input file format not recognized {l1c_path}, {l2c_path}, stop')
            raise RuntimeError from e

        for param in ['sza', 'vza', 'raa']:
            dc_l1c[param] = dc_l2c[param]
        del dc_l2c

        source_georeferencing = self.source_georeferencing(dc_l1c)

        # dc_l1c = dc_l1c.chunk({'x': 200, 'y': 200, 'wl': 10})

        if geoproject:
            dc_l1c = Reproj().regridding(dc_l1c, parallel=parallel)
        else:
            # Enforce the spectral dimension order in the output contract;
            # spatial-only variables remain on (y, x).
            dc_l1c = dc_l1c.transpose("wl", "y", "x", missing_dims="ignore")

        self.scene_metadata = self.make_scene_metadata(
            dc_l1c,
            platform='PRISMA',
            source_georeferencing=source_georeferencing,
            source_paths={'l1c': l1c_path, 'l2c': l2c_path},
        )

        return dc_l1c

    def read_l1c(self,
                 l1c_path: str,
                 reflectance_unit=False,
                 drop_vars=False):
        '''
        Load PRISMA L1C data into xarray rasters

        :param l1c_path: absolute path to the .h5 prisma file
        :param reflectance_unit: to convert from TOA radiance to TOA reflectance
        :param drop_vars: if True remove the radiance raster to keep reflectance only
        :return: xarray.Dataset with ``Ltoa`` and/or ``Rtoa`` (y, x, wl), ``F0``, ``fwhm``, ``lon``, ``lat``
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

    def read_l2c(self,
                 l2c_path: str):
        '''
        Load PRISMA L2C data (including observation angles) into xarray rasters

        :param l2c_path: absolute path to the .h5 prisma file
        :return: xarray.Dataset with the angles ``sza``, ``vza``, ``raa`` (and the other L2C variables)
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
