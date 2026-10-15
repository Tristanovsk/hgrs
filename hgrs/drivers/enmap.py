import os
import glob
import datetime as dt
import logging
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from scipy.interpolate import RegularGridInterpolator

from ..auxdata import SolarIrradiance
from ..utils import Misc
from ..spectral_sensitivity import Gaussian
from ..config import SensorDescription
from .base import L1Driver

logger = logging.getLogger(__name__)


class EnmapDriver(L1Driver):
    r'''
    Reader of the EnMAP L1C products (directory with the ``SPECTRAL_IMAGE`` raster and the ``METADATA`` XML).

    The TOA radiance :math:`L_{TOA}` (mW m\ :sup:`-2` sr\ :sup:`-1` nm\ :sup:`-1`) is converted into TOA
    reflectance

    .. math::

       R_{TOA}(\lambda) = \frac{\pi\, L_{TOA}(\lambda)}{F_0(\lambda)\, \cos\theta_s}

    where :math:`F_0` is the TSIS-1 solar irradiance (Coddington et al., 2021) corrected for the
    Earth-Sun distance (:py:meth:`~hgrs.utils.Misc.earth_sun_correction`) and convolved with the
    spectral response of the bands, and :math:`\theta_s` the mean solar zenith angle over the image.

    Examples
    --------
    .. code-block:: python

       from hgrs.drivers import EnmapDriver

       l1c = EnmapDriver().read('ENMAP01-____L1C-DT0000088121_..._V010502_...')
       l1c.Rtoa.sel(wl=[650, 560, 480], method='nearest').plot.imshow(rgb='wl', robust=True)

    References
    ----------
    * Coddington, O. M., Richard, E. C., Harber, D., et al. (2021). The TSIS-1 hybrid solar reference
      spectrum. *Geophysical Research Letters*, 48(12), e2020GL091709. https://doi.org/10.1029/2020GL091709
    * Storch, T., Honold, H.-P., Chabrillat, S., et al. (2023). The EnMAP imaging spectroscopy mission
      towards operations. *Remote Sensing of Environment*, 294, 113632.
      https://doi.org/10.1016/j.rse.2023.113632
    '''

    name = 'enmap'
    platform = 'EnMAP'

    def read(self,
             img_path: str,
             *,
             geoproject=True,
             reflectance_unit=True,
             drop_vars=False,
             filter_bad_bands=True
             ):
        '''
        Read an EnMAP L1C product.

        :param img_path: path of the L1C product directory
        :param geoproject: not used, the L1C image is already on a map grid (UTM)
        :param reflectance_unit: compute the TOA reflectance ``Rtoa``
        :param drop_vars: remove the radiance ``Ltoa`` (keep the reflectance only)
        :param filter_bad_bands: remove the bands of the VNIR-SWIR overlap
        '''
        l1c_path = img_path

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

        self.scene_metadata = self.make_scene_metadata(
            data,
            platform='EnMAP',
            source_georeferencing=self.source_georeferencing(data),
            source_paths={'l1c': l1c_path, 'metadata': metadata_path},
        )

        return data
