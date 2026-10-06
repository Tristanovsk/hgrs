import os

import numpy as np
import pandas as pd
import xarray as xr

from numba import jit
from scipy import ndimage

import matplotlib.pyplot as plt

from multiprocessing import Pool  # Process pool
from multiprocessing import sharedctypes
import itertools
from scipy.optimize import least_squares, minimize
from numba import njit, prange
import logging

from omnicloudmask import predict_from_array

from . import AuxData
from .config import behavior
from .lut_tables import LUTTables
from .spectral_sensitivity import SuperGaussian
from .config import (
    AerosolParameters,
    AppConfig,
    get_app_config,
    SceneMetadata,
    SensorDescription,
    WaterParameters,
)
opj = os.path.join


class Product():
    def __init__(self,
                 l1c_obj,
                 cams_data: xr.Dataset,

                 xcoarsen: int = None,
                 ycoarsen: int = None,
                 expon=2,
                 *,
                 app_config: AppConfig = None,
                 scene_metadata: SceneMetadata = None,
                 sensor_description: SensorDescription = None,
    ):

        self.config = get_app_config() if app_config is None else app_config
        self.atmospheric_correction = self.config.atmospheric_correction
        product_config = self.config.atmospheric_correction.product
        aerosol_config = self.config.atmospheric_correction.aerosol



        self.wl_to_remove = list(aerosol_config.excluded_wavelength_ranges_nm)
        self.wl_rgb = [30, 20, 10]

        # image chunking and coarsening parameters
        xcoarsen = product_config.x_coarsen if xcoarsen is None else xcoarsen
        ycoarsen = product_config.y_coarsen if ycoarsen is None else ycoarsen
        self.xcoarsen = xcoarsen
        self.ycoarsen = ycoarsen
        self.Npix_per_megapix = self.xcoarsen * self.ycoarsen
        self.block_size = product_config.block_size
        # minimum percentage of water pixel within the mega-pixel to enable processing
        self.pixel_percentage = product_config.water_pixel_percentage
        self.pixel_threshold = self.pixel_percentage / 100 * self.Npix_per_megapix

        # number of digits to keep for angle values
        self.ang_resol = product_config.angle_rounding_digits


        # CAMS-derived atmosphere auxiliary data
        self.cams_data = cams_data
        self.pressure = float(cams_data.sp) * 1e-2
        self.to3c = float(cams_data.gtco3)
        self.tno2c = float(cams_data.tcno2)
        self.tch4c = float(cams_data.tc_ch4)
        self.psl = 1013
        self.altitude = 0
        self.coef_abs_scat = product_config.gas_absorption_scattering_coefficient

        self.raster = l1c_obj.copy()

        # metadata objects
        self.scene_metadata = (
            scene_metadata
            if scene_metadata is not None
            else SceneMetadata.from_raster(self.raster)
        )
        lut_file = str(self.config.resolve_lut('toa'))
        trans_lut_file = str(self.config.resolve_lut('transmittance'))
        abs_gas_file = str(self.config.resolve_lut('abs_gas'))
        self.lut_tables = LUTTables(
            lut_file=lut_file,
            trans_lut_file=trans_lut_file,
            abs_gas_file=abs_gas_file,
        )
        self.aero_lut = self.lut_tables.aero_lut
        self.water_parameters = WaterParameters(
            mask=product_config,
            vapor=self.config.atmospheric_correction.water_vapor,
        )
        # Preserve the established Product parameter attributes for callers
        # that still access them directly; config objects remain authoritative.
        self.wl_water_vapor = self.water_parameters.wl_water_vapor
        self.wl_sunglint = self.water_parameters.wl_sunglint
        self.wl_green = self.water_parameters.wl_green
        self.wl_nir = self.water_parameters.wl_nir
        self.wl_1600 = self.water_parameters.wl_1600
        self.wl_atmo = list(aerosol_config.fit_wavelengths_nm)
        self.wl_non_neg = list(aerosol_config.nonnegative_reflectance_wavelengths_nm)
        self.sunglint_threshold = product_config.sunglint_threshold
        self.ndwi_threshold = product_config.ndwi_threshold
        self.green_swir_index_threshold = product_config.green_swir_index_threshold
        self.aerosol_parameters = AerosolParameters.from_cams(
            cams_data=cams_data,
            aerosol_lut=self.aero_lut,
            config=aerosol_config,
        )
        if sensor_description is None:
            sensor_description = SensorDescription.default(
                self.raster,
                name=self.scene_metadata.platform,
            )
        self.sensor_description = sensor_description

        self.fwhm = self.raster.fwhm.reset_coords(drop=True)  # .to_dataframe()
        self.wl = self.raster.wl
        self.sza_mean = np.nanmean(self.scene_metadata.solar_zenith)
        self.vza_mean = np.nanmean(self.scene_metadata.viewing_zenith)
        self.raa_mean = np.nanmean(self.scene_metadata.relative_azimuth)
        self.get_air_mass()

        self.Tg_other = None

        logging.info('Load pre-computed radiative transfer LUT')
        # pre-computed auxiliary data

        self.abs_gas_file = self.lut_tables.abs_gas_file
        self.lut_file = self.lut_tables.lut_file
        self.trans_lut_file = self.lut_tables.trans_lut_file

        self.Twv_lut = self.lut_tables.interp_twv(
            self.wl, self.sensor_description
        )

        self.load_auxiliary_data()


        logging.info(
            "OPAC model: %s", self.aerosol_parameters.aerosol_model
        )

        # spectral function for sensor response convolution
        # set the convolution module
        # Keep the legacy Product.spectral attribute for notebooks
        self.spectral = Spectral(self.wl, self.fwhm.values)


    def load_auxiliary_data(self):
        # get hgrs auxdata
        self.auxdata = AuxData(self.wl)

    def return_dictionary(self):
        """Return the Product values written to output metadata and variables."""
        aerosol = self.aerosol_parameters
        water = self.water_parameters
        return {
            "wl_water_vapor": water.wl_water_vapor,
            "wl_sunglint": water.wl_sunglint,
            "wl_atmo": list(aerosol.fit_wavelengths_nm),
            "wl_to_remove": self.wl_to_remove,
            "wl_non_neg": list(aerosol.nonnegative_reflectance_wavelengths_nm),
            "wl_green": water.wl_green,
            "wl_nir": water.wl_nir,
            "wl_1600": water.wl_1600,
            "wl_rgb": self.wl_rgb,
            "dirdata": str(self.config.data_root),
            "xcoarsen": self.xcoarsen,
            "ycoarsen": self.ycoarsen,
            "Npix_per_megapix": self.Npix_per_megapix,
            "block_size": self.block_size,
            "pixel_percentage": self.pixel_percentage,
            "pixel_threshold": self.pixel_threshold,
            "ang_resol": self.ang_resol,
            "abs_gas_file": self.lut_tables.abs_gas_file,
            "lut_file": self.lut_tables.lut_file,
            "water_vapor_transmittance_file": self.lut_tables.water_vapor_transmittance_file,
            "sunglint_threshold": water.mask.sunglint_threshold,
            "ndwi_threshold": water.mask.ndwi_threshold,
            "green_swir_index_threshold": water.mask.green_swir_index_threshold,
            "pressure": self.pressure,
            "to3c": self.to3c,
            "tno2c": self.tno2c,
            "tch4c": self.tch4c,
            "psl": self.psl,
            "altitude": self.altitude,
            "coef_abs_scat": self.coef_abs_scat,
            "water_vapor_smoothing_method": water.vapor.smoothing_method,
            "aerosol_smoothing_method": aerosol.smoothing_method,
            "aerosol_solver_max_iterations": aerosol.solver_max_iterations,
            "aerosol_solver_ftol": aerosol.solver_ftol,
            "aerosol_solver_warm_start": aerosol.solver_warm_start,
            "bidirectional_transmittance_global": (
                self.atmospheric_correction.bidirectional_transmittance_global
            ),
        }

    def apply_water_masks(self):
        green = self.raster.Rtoa.sel(wl=self.water_parameters.wl_green).mean(dim='wl')
        nir = self.raster.Rtoa.sel(wl=self.water_parameters.wl_nir).mean(dim='wl')
        ndwi = (green - nir) / (green + nir)

        green = self.raster.Rtoa.sel(wl=self.water_parameters.wl_green).mean(dim='wl')
        b1600 = self.raster.Rtoa.sel(wl=self.water_parameters.wl_1600).mean(dim='wl')
        green_swir_index = (green - b1600) / (green + b1600)
        b2200 = self.raster.Rtoa.sel(wl=self.water_parameters.wl_sunglint).mean(dim='wl')
        water_mask = (
            (ndwi > self.water_parameters.mask.ndwi_threshold)
            & (b2200 < self.water_parameters.mask.sunglint_threshold)
            & (green_swir_index > self.water_parameters.mask.green_swir_index_threshold)
        )
        self.raster['water_mask'] = water_mask.rename('water_mask')
        self.raster['water_mask'].attrs.update(
            long_name='spectral water mask',
            description='True where the pixel passes all configured spectral water tests',
        )
        self.raster['Rtoa'] = self.raster.Rtoa.where(
            water_mask
        ).load()

    def apply_land_mask(self):
        """Mask PRISMA pixels classified as land; leave other sensors unchanged."""
        product_config = self.atmospheric_correction.product
        if not product_config.prisma_land_mask_enabled:
            return
        if self.scene_metadata.platform.upper() != "PRISMA":
            return
        if "landcover_mask" not in self.raster:
            return

        from scipy.ndimage import binary_dilation

        classes = self.raster.landcover_mask.values
        valid = np.isin(classes, product_config.prisma_land_valid_classes)
        forbidden = np.isin(classes, product_config.prisma_land_forbidden_classes)
        # Preserve unlabeled pixels away from known land, as in the sibling path.
        unlabeled_water = (classes == 255) & ~binary_dilation(forbidden)
        land_mask = valid | unlabeled_water
        self.raster["Rtoa"] = self.raster.Rtoa.where(land_mask)
        self.land_mask = xr.DataArray(
            land_mask,
            dims=("y", "x"),
            coords={"y": self.raster.y, "x": self.raster.x},
            name="land_mask",
            attrs={"description": "PRISMA land-cover water mask"},
        )

    def get_omnicloudmask(self,
                          rgnir):

        '''
        Apply OmniCloudMAsk for clouds and cloud shadows masking

        Outputs:
            0 = Clear
            1 = Thick Cloud
            2 = Thin Cloud
            3 = Cloud Shadow

        see https://github.com/DPIRD-DMA/OmniCloudMask

        refs:
         Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., & George, R. J. (2025).
         Training sensor-agnostic deep learning models for remote sensing:
         Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask.
         Remote Sensing of Environment, 322, 114694. https://doi.org/10.1016/J.RSE.2025.114694

        :param rgnir: raster xarray object with the red, green and nir bands
        :return omnimask: raster of the retrieved mask
        '''

        pred = predict_from_array(rgnir.fillna(0).values)
        omnimask = xr.DataArray(pred[0],
                                dims=["y", "x"],
                                coords=dict(x=rgnir.x.values,
                                            y=rgnir.y.values,
                                            time=rgnir.time,
                                            ),
                                attrs=dict(
                                    description="OmniCloudMask, see https://github.com/DPIRD-DMA/OmniCloudMask",
                                    reference='https://doi.org/10.1016/J.RSE.2025.114694'),
                                )
        omnimask.name = 'omnimask'
        return omnimask

    def round_angles(self):
        for param in ['sza', 'vza', 'raa']:
            self.raster[param] = self.raster[param].round(self.ang_resol)

    def get_air_mass(self, raster_name='raster', round=True, digit_resol=3):
        raster = self.__dict__[raster_name]
        raster['air_mass'] = 1. / np.cos(np.radians(raster.sza)) + 1. / np.cos(np.radians(raster.vza))
        if round:
            raster['air_mass'] = raster['air_mass'].round(digit_resol)
        self.air_mass_mean = np.nanmean(raster['air_mass'].values)

    @staticmethod
    def remove_wl_dataarray(xarr, wl_to_remove, drop=True):
        xarr_ = xarr.isel(x=1, y=1)
        for wls in wl_to_remove:
            wl_min, wl_max = wls
            xarr_ = xarr_.where((xarr_.wl < wl_min) | (xarr_.wl > wl_max), drop=drop)
        wl_final = xarr_.wl.values
        return xarr.sel(wl=wl_final)

    @staticmethod
    def remove_wl_dataset(xds, wl_to_remove, variable='Rtoa', drop=True):
        xarr_ = xds[variable].isel(x=1, y=1)
        for wls in wl_to_remove:
            wl_min, wl_max = wls
            xarr_ = xarr_.where((xarr_.wl < wl_min) | (xarr_.wl > wl_max), drop=drop)
        wl_final = xarr_.wl.values
        return xds.sel(wl=wl_final)

    def plot_angles(self, raster_name='raster',
                    figsize=(20, 4),
                    cmap=plt.cm.Spectral_r, **kwargs):
        raster = self.__dict__[raster_name]
        params = [raster.sza, raster.vza, raster.raa, raster.air_mass]
        titles = ['SZA', 'VZA', 'rel. AZI', 'Air mass']
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize)

        for i, ax in enumerate(axs):
            params[i].plot.imshow(ax=ax, robust=True, cmap=cmap, **kwargs)
            ax.set_title(titles[i])
            ax.set(xticks=[], yticks=[])
            ax.set_ylabel('')
            ax.set_xlabel('')
        return fig

    def plot_params(self, xds,
                    params=['aot_ref', 'aot_ref_std', 'brdfg', 'brdfg_std'],
                    shrink=0.8,
                    cmap=plt.cm.Spectral_r):

        ncols = len(params)
        fig_width = ncols * 5 + 2
        fig, axs = plt.subplots(1, ncols=ncols, figsize=(fig_width, 4))
        axs = axs.ravel()

        for i in range(4):
            xds[params[i]].plot.imshow(cmap=cmap, robust=True, vmin=0,  # vmax=0.201,
                                       cbar_kwargs={'shrink': shrink, 'label': params[i]},
                                       ax=axs[i])  # extent=extent_val, transform=proj,
            axs[i].set(xticks=[], yticks=[])
            axs[i].set_ylabel('')
            axs[i].set_xlabel('')
            axs[i].set_title(params[i])
        return fig

    def plot_masks(self, params=['cloud_mask', 'sunglint_mask', 'landcover_mask'],
                   vmax=12,
                   shrink=0.8,
                   cmap=plt.cm.Spectral_r):

        ncols = len(params)
        fig_width = ncols * 5 + 1
        fig, axs = plt.subplots(1, ncols=ncols, figsize=(fig_width, 4))

        fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.99,
                            hspace=0.15, wspace=0.15)
        axs = axs.ravel()

        for i, param in enumerate(params):
            self.raster[param].plot.imshow(cmap=cmap, vmax=vmax, robust=True,
                                           cbar_kwargs={'shrink': shrink},
                                           ax=axs[i])  # extent=extent_val, transform=proj,
            axs[i].set(xticks=[], yticks=[])
            axs[i].set_ylabel('')
            axs[i].set_xlabel('')
            axs[i].set_title(param)
        return fig

    def rgb(self, variable='Rtoa', raster_name='raster', gamma=0.5, brightness_factor=1, **kwargs):
        fig = (self.__dict__[raster_name][variable].isel(
            wl=self.wl_rgb) ** gamma * brightness_factor).plot.imshow(rgb='wl', robust=True, **kwargs)
        fig.axes.set(xticks=[], yticks=[])
        fig.axes.set_ylabel('')
        fig.axes.set_xlabel('')
        return fig

    def plot_water_pix_number(self, cmap=plt.cm.Spectral_r, **kwargs):
        try:
            fig = self.coarse_masked_raster['water_pixel_number'].plot.imshow(cmap=cmap, robust=True, **kwargs)
            fig.axes.set(xticks=[], yticks=[])
            fig.axes.set_ylabel('')
            fig.axes.set_xlabel('')
            return fig
        except:
            print('please apply algo.get_coarse_masked_raster() before')


class Algo(Product):

    def __init__(
        self,
        l1c_obj,
        cams_data: xr.Dataset,
        xcoarsen: int = None,
        ycoarsen: int = None,
        expon=2,
        *,
        app_config: AppConfig = None,
        scene_metadata: SceneMetadata = None,
        sensor_description: SensorDescription = None,
    ):
        Product.__init__(
            self,
            l1c_obj,
            cams_data,
            xcoarsen,
            ycoarsen,
            expon,
            app_config=app_config,
            scene_metadata=scene_metadata,
            sensor_description=sensor_description,
        )

    def get_pressure(self, alt, psl):
        '''Compute the pressure for a given altitude
           alt : altitude in meters (float or np.array)
           psl : pressure at sea level in hPa
           palt : pressure at the given altitude in hPa'''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    def get_coarse_raster(self, variables=['sza', 'vza', 'raa', 'air_mass', 'Rtoa']):
        self.coarse_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen, boundary="pad").mean()

    def get_coarse_masked_raster(self, variables=['sza', 'vza', 'raa', 'air_mass', 'Rtoa']):
        self.coarse_masked_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen,
                                                                   boundary="pad").mean()
        self.coarse_masked_raster['water_pixel_number'] = self.raster['Rtoa']. \
            isel(wl=slice(10, 20)).mean(dim='wl'). \
            coarsen(x=self.xcoarsen, y=self.ycoarsen, boundary="pad").count()

    def get_gaseous_optical_thickness(self):
        gas_lut = self.lut_tables.gas_lut
        ot_o3 = gas_lut.o3 * self.to3c
        ot_ch4 = gas_lut.ch4 * self.tch4c
        ot_no2 = gas_lut.no2 * self.tno2c
        ot_air = (gas_lut.co + self.coef_abs_scat * gas_lut.co2 +
                  self.coef_abs_scat * gas_lut.o2 +
                  self.coef_abs_scat * gas_lut.o4) * self.pressure / 1000
        self.abs_gas_opt_thick = ot_ch4 + ot_no2 + ot_o3 + ot_air

    def get_gaseous_transmittance(self):

        self.get_gaseous_optical_thickness()
        Tg = np.exp(- self.air_mass_mean * self.abs_gas_opt_thick)

        self.Tg_other = self.sensor_description.convolve(Tg).rename({"wl_sensor": "wl"})

        # fwhms = self.raster.fwhm.reset_coords(drop=True).to_dataframe()
        # Tg_int = []
        # for mu, fwhm in fwhms.iterrows():
        #     sig = self.Gamma2sigma(fwhm.values)
        #     rsr = self.gaussian(wl_ref, mu, sig)
        #     Tg_ = (Tg * rsr).integrate('wl') / np.trapezoid(rsr, wl_ref)
        #     Tg_int.append(Tg_.values)
        #
        # self.Tg_other = xr.DataArray(Tg_int, name='Ttot', coords={'wl': self.raster.wl.values})

    def other_gas_correction(self, raster_name='coarse_masked_raster', variable='Rtoa'):
        raster = self.__dict__[raster_name]
        attrs = raster[variable].attrs
        if attrs.__contains__('other_gas_correction'):
            if attrs['other_gas_correction']:
                print('raster ' + raster_name + '.' + variable + ' is already corrected for other gases transmittance')
                print('set attribute other_gas_correction to False to proceed anyway')
                return
        if self.Tg_other is None:
            self.get_gaseous_transmittance(self.air_mass_mean)
        raster[variable] = raster[variable] / self.Tg_other
        raster[variable].attrs['other_gas_correction'] = True

    def water_vapor_correction(self,
                               raster_name='coarse_masked_raster',
                               variable='Rtoa'):
        '''

        :param raster_name:
        :param variable:
        :return:
        '''

        raster = self.__dict__[raster_name]
        attrs = raster[variable].attrs
        if attrs.__contains__('water_vapor_correction'):
            if attrs['other_gas_correction']:
                print('raster ' + raster_name + '.' + variable + ' is already corrected for water vapor transmittance')
                print('set attribute other_gas_correction to False to proceed anyway')
                return

        if self.Twv_raster is None:
            print('xarray of water vapor transmittance is not set, please run get_wv_transmittance_raster(tcwv_raster)')
            return

        raster[variable] = raster[variable] / self.Twv_raster
        raster[variable].attrs['water_vapor_correction'] = True

    def get_wv_transmittance_raster(self, tcwv_raster):
        tcwv_source = tcwv_raster.get("tcwv_smooth", tcwv_raster.tcwv)
        tcwv_vals = tcwv_source.round(1)
        tcwvs = np.unique(tcwv_vals)
        tcwvs = tcwvs[~np.isnan(tcwvs)]
        # TODO improve for air_mass raster
        Twvs = self.Twv_lut.Twv.interp(air_mass=self.air_mass_mean).interp(tcwv=tcwvs, method='linear').drop('air_mass')
        self.Twv_raster = Twvs.interp(tcwv=tcwv_vals, method='nearest')
        if not behavior.PREVIOUS_BEHAVIOR:
            self.Twv_raster = self.Twv_raster.where(tcwv_vals.notnull())
        # The lookup target is an auxiliary coordinate, not a second TCWV
        # product variable. Keep it from propagating onto Rtoa/Rrs, where it
        # conflicts with the retrieved coarse-grid ``tcwv`` output variable.
        if 'tcwv' in self.Twv_raster.coords:
            self.Twv_raster = self.Twv_raster.drop_vars('tcwv')

    def get_full_resolution(self, xarr):
        return xarr.interp(x=self.raster.x, y=self.raster.y)


class Solver():
    def __init__(self):
        pass

    def errFit(self, hess_inv, resVariance):
        '''
        Error/uncertainty of the estimated parameters
        :param resVariance:
        :return:
        '''
        return np.sqrt(np.diag(hess_inv * resVariance))

    @staticmethod
    def fill_na_conv(values):
        """Keep a valid center cell; replace a NaN center by its neighbors' mean."""
        center = len(values) // 2
        if not np.isnan(values[center]):
            return values[center]
        neighbors = np.delete(values, center)
        if np.isnan(neighbors).all():
            return np.nan
        return np.nanmean(neighbors)

    def fill_nodata_local(self, values, footprint):
        """Fill only NaN cells from a local nan-mean neighborhood."""
        values = np.asarray(values, dtype=float)
        if not np.isnan(values).any():
            return values.copy()
        return ndimage.generic_filter(
            values,
            function=self.fill_na_conv,
            footprint=np.ones(footprint, dtype=bool),
            mode="constant",
            cval=np.nan,
        )

    def conv_mapping(self, x):
        """
        Nan-mean convolution
        """
        # get index of central pixel
        idx = len(x) // 2
        if np.isnan(x[idx]) and not np.isnan(np.delete(x, idx)).all():
            return np.nanmean(np.delete(x, idx))
        elif np.isnan(np.delete(x, idx)).all():
            return x[idx]
        else:
            return np.nanmean(x)

    @staticmethod
    @jit(nopython=True)
    def filter2d(image, weight, windows):
        '''
         Function to convolve parameter image with uncertainty image
        :param image: parameter image
        :param weight: uncertainty image
        :param windows: size of the window for convolution
        :return: convolved result with same shape as image

        '''
        M, N = np.shape(image)
        Mf, Nf = windows
        Mf2 = Mf // 2
        Nf2 = Nf // 2
        threshold = 0
        result = image
        for i in range(M):
            for j in range(N):
                num = 0.0
                norm = 0.0
                if weight[i, j] > threshold:
                    for ii in range(Mf):
                        ix = i - Mf2 + ii
                        if ix < M:
                            for jj in range(Nf):

                                iy = j - Nf2 + jj
                                if iy < N:
                                    wgt = weight[ix, iy]
                                    if wgt > 0.:
                                        num += (wgt * image[ix, iy])
                                        norm += wgt
                    result[i, j] = num / norm
        return result


class WaterVapor(Solver):

    def __init__(self, prod,
                 raster_name='coarse_masked_raster',
                 variable='Rtoa'):

        self.prod = prod
        self.raster = prod.__dict__[raster_name]
        self.air_mass = prod.air_mass_mean
        # get data for the subset of "water vapor" wavelengths
        self.parameters = prod.water_parameters.vapor
        data = self.raster[variable].sel(wl=prod.water_parameters.wl_water_vapor)
        self.data = data
        self.nwl, self.height, self.width = data.shape
        self.x = data.x
        self.y = data.y
        self.wl = data.wl

        # TODO improve to process the air mass raster instead of scalar mean value
        # TODO check impact of method = 'nearest'
        self.Twv_ = prod.lut_tables.interp_twv(
            self.data.wl, prod.sensor_description
        ).Twv.interp(air_mass=self.air_mass)
        self.Twv_['wl'] = self.Twv_['wl'] / 1000
        self.wl_mic = self.Twv_.wl.values

    def toa_simu(self, wl, Twv, tcwv, a, b):
        '''wl in micron
        '''
        # print(Twv.tcwv)
        return Twv.interp(tcwv=tcwv, method='linear').values * (a * wl + b)

    def toa_simu2(self, wl, Twv, tcwv, c0, c1, c2, c3):

        return c0 * np.exp(-c1 * wl ** -c2) * self.Twv_.interp(tcwv=tcwv).values \
            + c3 * self.wl_ ** -3 * self.Twv_.interp(tcwv=0.3 * tcwv).values

    def func(self, x, Twv, wl, y):
        return self.toa_simu(wl, Twv, *x) - y

    def func2(self, x, Twv, wl, y):
        return self.toa_simu2(wl, Twv, *x) - y

    def solve(self, x0=None):
        if x0 is None:
            x0 = self.parameters.first_guess

        result = np.ctypeslib.as_ctypes(np.full((self.width, self.height, 6), np.nan))
        shared_array = sharedctypes.RawArray(result._type_, result)
        self.x0 = x0
        data = self.data
        height = self.height
        width = self.width
        block_size = self.prod.block_size
        pixel_threshold = self.prod.pixel_threshold
        if list(self.raster.keys()).__contains__('water_pixel_number'):
            water_pixel_number = self.raster.water_pixel_number
        else:
            water_pixel_number = None

        global chunk_process

        def chunk_process(args):

            window_x, window_y = args

            tmp = np.ctypeslib.as_array(shared_array)
            # x0 = [20, -0.04, 0.1]
            for ix in range(window_x, min(width, window_x + block_size)):
                for iy in range(window_y, min(height, window_y + block_size)):
                    if water_pixel_number is not None:
                        if water_pixel_number.isel(x=ix, y=iy).values < pixel_threshold:
                            continue

                    y = data.isel(x=ix, y=iy).dropna(dim='wl')
                    # sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')
                    # TODO put solver parameter in self instance
                    res_lsq = least_squares(self.func, self.x0, args=(self.Twv_, self.wl_mic, y),
                                            bounds=([0, -10, 0], [60, 1, 1]),
                                            diff_step=1e-2, xtol=1e-2, ftol=1e-2, max_nfev=20)
                    xres = res_lsq.x
                    resVariance = (res_lsq.fun ** 2).sum() / (len(res_lsq.fun) - len(res_lsq.x))
                    hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
                    try:
                        hess_inv = np.linalg.inv(hess)
                        std = self.errFit(hess_inv, resVariance)
                    except:
                        std = [np.nan, np.nan, np.nan]
                    tmp[ix, iy, :] = [*xres, *std]
            return

        window_idxs = [(i, j) for i, j in
                       itertools.product(range(0, width, block_size),
                                         range(0, height, block_size))]

        p = Pool()
        res = p.map(chunk_process, window_idxs)
        result = np.ctypeslib.as_array(shared_array)
        self.result = result
        self.water_vapor = xr.Dataset(dict(tcwv=(["y", "x"], result[:, :, 0].T),
                                           tcwv_std=(["y", "x"], result[:, :, 3].T)),
                                      coords=dict(
                                          x=self.x,
                                          y=self.y),
                                      attrs=dict(
                                          description="Fitted Total Columnar Water vapor; warning for transmittance computation only",
                                          units="kg/m**2")
                                      )

        if not behavior.PREVIOUS_BEHAVIOR:
            valid_tcwv = self.water_vapor.tcwv.notnull()
            self.water_vapor["water_vapor_validity_mask"] = valid_tcwv.astype(
                np.uint8
            )
            self.water_vapor["water_vapor_validity_mask"].attrs.update(
                long_name="water vapor retrieval validity mask",
                description=(
                    "1 where a finite TCWV retrieval was available before spatial "
                    "interpolation; 0 where the retrieval was missing"
                ),
                flag_values=np.array([0, 1], dtype=np.uint8),
                flag_meanings="invalid valid",
            )

        settings = self.parameters
        if settings.smoothing_method != "none":
            values = self.water_vapor.tcwv.values.astype(float)
            if settings.smoothing_method == "weighted_local":
                std = self.water_vapor.tcwv_std.values.astype(float)
                values = self.filter2d(
                    values, 1.0 / std**2, settings.smoothing_window
                )
            elif settings.smoothing_method != "nanmean_local":
                raise ValueError(
                    f"Unknown water-vapor smoothing method: {settings.smoothing_method}"
                )
            values = ndimage.generic_filter(
                values,
                function=self.conv_mapping,
                footprint=np.ones(settings.smoothing_window, dtype=bool),
                mode="nearest",
            )
            self.water_vapor["tcwv_smooth"] = (("y", "x"), values)

        if not behavior.PREVIOUS_BEHAVIOR:
            self.water_vapor["tcwv"] = (
                ("y", "x"),
                self.fill_nodata_local(
                    self.water_vapor.tcwv.values, settings.smoothing_window
                ),
            )
            if "tcwv_smooth" in self.water_vapor:
                self.water_vapor["tcwv_smooth"] = (
                    ("y", "x"),
                    self.fill_nodata_local(
                        self.water_vapor.tcwv_smooth.values,
                        settings.smoothing_window,
                    ),
                )


class Aerosol(Solver):

    def __init__(self, prod,
                 raster_name='coarse_masked_raster',
                 variable='Rtoa'):

        self.prod = prod
        self.sensor_description = prod.sensor_description
        aerosol_state = prod.aerosol_parameters
        self.aerosol_state = aerosol_state
        self.aerosol_model = aerosol_state.aerosol_model
        self.raster = prod.__dict__[raster_name]
        self.auxdata = prod.auxdata
        self.aero_lut = prod.aero_lut

        # set box limits in aod550 for non-linear optimization
        self.aod550_min = aerosol_state.aod550_min
        self.aod550_max = aerosol_state.aod550_max
        self.first_guess = [aerosol_state.aod550_initial, 0.]

        # get full resolution parameters
        self.xfull = prod.raster.x
        self.yfull = prod.raster.y

        # get data for the subset of "black water" wavelengths
        self.data = self.raster[variable]
        data = self.data
        self.wl_atmo = list(aerosol_state.fit_wavelengths_nm)
        self.wl_non_neg = list(aerosol_state.nonnegative_reflectance_wavelengths_nm)
        self.nwl, self.height, self.width = data.shape
        self.x = data.x
        self.y = data.y
        self.wl = data.wl

        self.sza = prod.sza_mean
        self.vza = prod.vza_mean
        self.raa = prod.raa_mean
        self.raa_lut = (180 - self.raa) % 360
        self.air_mass = prod.air_mass_mean
        self.pressure = prod.pressure

        # process parameters
        self.block_size = self.prod.block_size
        self.pixel_threshold = self.prod.pixel_threshold
        self.wl_sunglint = self.prod.water_parameters.wl_sunglint
        self.prepare_lut(self.wl)

    def prepare_lut(self, wl):

        auxdata = self.auxdata
        sza = self.sza
        vza = self.vza
        raa_lut = self.raa_lut

        self.sunglint_eps = auxdata.sunglint_eps.interp(wl=wl)
        self.rot = auxdata.rot.interp(wl=wl) * self.pressure / self.auxdata.pressure_rot_ref

        aot_refs = [0, *np.logspace(-3, np.log10(0.8), 100)]
        self.aot_lut = (
            self.aero_lut
            .sel(model=self.aerosol_model)
            .aot.interp(wl=wl,method="quadratic")
            .interp(aot_ref=aot_refs, method='quadratic').dropna('aot_ref')
        )

        norm_radiance = (
            self.aero_lut
            .sel(model=self.aerosol_model).I
            .interp(vza=vza, azi=raa_lut, method='linear')
            .interp(sza=sza, method='quadratic').squeeze()
        )
        self.Rtoa_lut = (
            norm_radiance.interp(wl=wl, method="quadratic")
            .interp(aot_ref=aot_refs, method='quadratic')
            .dropna('aot_ref') / np.cos(np.radians(sza))
        )

    def transmittance_dir(self, aot, M, rot=0):
        return np.exp(-(rot + aot) * M)

    def toa_simu(self, aot, rot, Rtoa_lut, sunglint_eps, aot_ref, BRDFg):
        '''
        '''
        aot = aot.interp(aot_ref=aot_ref)
        Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
        Tdir = self.transmittance_dir(aot, self.air_mass, rot=rot)
        sunglint_corr = Tdir * sunglint_eps
        Rdir = sunglint_corr * BRDFg / (Tdir.sel(wl=self.wl_sunglint) * sunglint_eps.sel(wl=self.wl_sunglint)).mean(
            dim='wl')
        # sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])

        return Rdiff + Rdir

    def func(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        return (y - self.toa_simu(aot, rot, Rtoa_lut, sunglint_eps, *x))  # /sigma

    def cost_func(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        return np.sum((self.func(x, aot, rot, Rtoa_lut, sunglint_eps, y) ** 2))

    def constraint(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        return np.min((self.func(x, aot, rot, Rtoa_lut, sunglint_eps, y)))

    def solve(self, x0=[0.005, 0.]):

        result = np.ctypeslib.as_ctypes(np.full((self.width, self.height, 4), np.nan))
        shared_array = sharedctypes.RawArray(result._type_, result)
        # TODO clean up method to assign first guess
        self.x0 = x0
        self.x0 = self.first_guess

        data = self.data
        height = self.height
        width = self.width

        if list(self.raster.keys()).__contains__('water_pixel_number'):
            water_pixel_number = self.raster.water_pixel_number
        else:
            water_pixel_number = None

        global chunk_process

        def chunk_process(args):
            window_x, window_y = args
            tmp = np.ctypeslib.as_array(shared_array)
            x0 = self.x0
            for ix in range(window_x, min(width, window_x + self.block_size)):
                for iy in range(window_y, min(height, window_y + self.block_size)):
                    if water_pixel_number is not None:
                        if water_pixel_number.isel(x=ix, y=iy).values < self.pixel_threshold:
                            continue
                    # x0 = self.x0
                    yfull = data.isel(x=ix, y=iy).dropna(dim='wl')
                    # Pixels can pass the coarse water-count threshold while
                    # still having no valid spectrum after masking. xarray's
                    # nearest selection raises on an empty wavelength index;
                    # leave this pixel's preinitialized retrieval values NaN.
                    if yfull.sizes['wl'] == 0:
                        continue
                    # sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')

                    cons = ({'type': 'ineq',
                             'fun': self.constraint,
                             'args': (self.aot_lut, self.rot, self.Rtoa_lut, self.sunglint_eps,
                                      yfull.sel(wl=self.wl_non_neg, method='nearest'))
                             })

                    min_res = minimize(self.cost_func, x0,
                                       args=(self.aot_lut, self.rot, self.Rtoa_lut, self.sunglint_eps,
                                             yfull.sel(wl=self.wl_atmo, method='nearest')),
                                       method='SLSQP',
                                       bounds=((self.aod550_min, self.aod550_max), (0, 1.3)),
                                       constraints=cons, options=self._optimizer_options()
                                       )
                    xres = min_res.x
                    if min_res.success and self.aerosol_state.solver_warm_start:
                        x0 = xres
                    # except:
                    # print(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_, y)
                    #    break

                    std = [min_res.fun, np.sum(min_res.jac ** 2)]
                    tmp[ix, iy, :] = [*xres, *std]

        window_idxs = [(i, j) for i, j in
                       itertools.product(range(0, self.width, self.block_size),
                                         range(0, self.height, self.block_size))]

        p = Pool()
        res = p.map(chunk_process, window_idxs)
        result = np.ctypeslib.as_array(shared_array)

        self.aero_img = xr.Dataset(
            dict(
                aot_ref=(["y", "x"], result[:, :, 0].T),
                brdfg=(["y", "x"], result[:, :, 1].T),
                aot_ref_std=(["y", "x"], result[:, :, 2].T),
                brdfg_std=(["y", "x"], result[:, :, 3].T)
            ),
            coords=dict(
                x=self.x,
                y=self.y),
            attrs=dict(
                description="aerosol and sunglint retrieval from coarse resolution data",
                aerosol_model=self.aerosol_model
                )
        )
        if not behavior.PREVIOUS_BEHAVIOR:
            valid_aot_retrieval = self.aero_img.aot_ref.notnull()
            self.aero_img['aerosol_retrieval_validity_mask'] = (
                valid_aot_retrieval.astype(np.uint8)
            )
            self.aero_img['aerosol_retrieval_validity_mask'].attrs.update(
                long_name='aerosol retrieval validity mask',
                description=(
                    '1 where a finite AOT retrieval was available before spatial '
                    'interpolation; 0 where the retrieval was missing'
                ),
                flag_values=np.array([0, 1], dtype=np.uint8),
                flag_meanings='invalid valid',
            )
            if water_pixel_number is not None:
                self.aero_img['water_validity_mask'] = (
                    water_pixel_number >= self.pixel_threshold
                ).astype(np.uint8)
                self.aero_img['water_validity_mask'].attrs.update(
                    long_name='water pixel threshold validity mask',
                    description=(
                        '1 where water_pixel_number meets the aerosol retrieval '
                        'pixel threshold; 0 otherwise'
                    ),
                    flag_values=np.array([0, 1], dtype=np.uint8),
                    flag_meanings='invalid valid',
                )

            self.aero_img['aerosol_validity_mask'] = (
                data.notnull().any(dim='wl').astype(np.uint8)
            )
            self.aero_img['aerosol_validity_mask'].attrs.update(
                long_name='aerosol retrieval input validity mask',
                description=(
                    '1 where at least one aerosol retrieval wavelength is finite '
                    'after masking and gas-transmittance filtering; 0 otherwise. '
                    'This indicates input-spectrum availability, not optimizer success.'
                ),
                flag_values=np.array([0, 1], dtype=np.uint8),
                flag_meanings='invalid valid',
            )

    def _optimizer_options(self):
        options = {"maxiter": self.aerosol_state.solver_max_iterations}
        if self.aerosol_state.solver_ftol is not None:
            options["ftol"] = self.aerosol_state.solver_ftol
        return options

    def smoothing(self, weights, windows=None, mask=None):
        method = self.aerosol_state.smoothing_method
        param = self.aero_img['aot_ref'].__deepcopy__().to_numpy().astype(float)
        if method == "none":
            res = param
        elif method == "weighted_local":
            windows = self.aerosol_state.smoothing_window if windows is None else windows
            footprint = self.aerosol_state.smoothing_footprint if mask is None else mask
            aot_ref_smoothed = self.filter2d(param, weights, windows)
            res = ndimage.generic_filter(
                aot_ref_smoothed,
                function=self.conv_mapping,
                footprint=np.ones(footprint, dtype=bool),
                mode='nearest',
            )
        elif method == "nanmean_local":
            footprint = self.aerosol_state.smoothing_footprint if mask is None else mask
            res = ndimage.generic_filter(
                param, function=self.conv_mapping,
                footprint=np.ones(footprint, dtype=bool), mode='nearest'
            )
        else:
            raise ValueError(f"Unknown aerosol smoothing method: {method}")

        self.aero_img['aot_ref_smoothed'] = xr.DataArray(res, coords=dict(y=self.aero_img.y, x=self.aero_img.x))

    def get_aot_full_resolution(self):
        # TODO change fill_value (extrapolate is not safe) with median of retrieval, for instance
        self.aot_ref_full = \
            self.aero_img['aot_ref_smoothed'].interp(x=self.xfull, y=self.yfull,
                                                     method='linear', kwargs={"fill_value": "extrapolate"})

    def get_atmo_parameters(self,
                            prod):


        wl = prod.coarse_masked_raster.wl
        weights = prod.coarse_masked_raster['water_pixel_number'].__deepcopy__().to_numpy().astype(float)

        # get LUT for desired wavelengths
        self.prepare_lut(wl)
        self.smoothing(weights)
        if not behavior.PREVIOUS_BEHAVIOR:
            self.aero_img['aot_ref'] = (
                ('y', 'x'),
                self.fill_nodata_local(
                    self.aero_img.aot_ref.values,
                    self.aerosol_state.smoothing_footprint,
                ),
            )
            self.aero_img['aot_ref_smoothed'] = (
                ('y', 'x'),
                self.fill_nodata_local(
                    self.aero_img.aot_ref_smoothed.values,
                    self.aerosol_state.smoothing_footprint,
                ),
            )
        self.get_aot_full_resolution()

        # construct aot raster
        aot_ref_source = self.aero_img['aot_ref_smoothed']
        if behavior.PREVIOUS_BEHAVIOR:
            aot_ref_median = aot_ref_source.median()
            aot_ref_vals = aot_ref_source.fillna(aot_ref_median).round(3)
        else:
            aot_ref_vals = aot_ref_source.round(3)
        valid_aot = aot_ref_vals.notnull()
        aot_refs = np.unique(aot_ref_vals)
        aot_refs = aot_refs[~np.isnan(aot_refs)]
        # TODO update LUT for aot< 0.001
        aot_refs[aot_refs < 0.002] = 0.002

        if len(aot_refs) == 0:
            aots = self.aot_lut.isel(aot_ref=0, drop=True) + aot_ref_vals
            Rdiffs = self.Rtoa_lut.isel(aot_ref=0, drop=True) + aot_ref_vals
        else:
            # if rounded aot_ref has unique value
            if len(aot_refs) == 1:
                aot_refs = np.concatenate([aot_refs, 1.2 * aot_refs])
            aots = self.aot_lut.interp(aot_ref=aot_refs, method='linear')
            aots = aots.interp(aot_ref=aot_ref_vals, method='nearest')
            Rdiffs = self.Rtoa_lut.interp(aot_ref=aot_refs, method='linear')
            Rdiffs = Rdiffs.interp(aot_ref=aot_ref_vals, method='nearest')
            if not behavior.PREVIOUS_BEHAVIOR:
                aots = aots.where(valid_aot)
                Rdiffs = Rdiffs.where(valid_aot)

        aots.name = 'aot'
        aots.attrs['description'] = 'spectral aerosol optical thickness'

        # construct raster for diffuse atmospheric reflectance
        Rdiffs.name = 'Rtoa_diff'
        Rdiffs.attrs['description'] = 'top-of-atmosphere atmosphere reflectance'

        # construct raster for direct transmittance due to rayleigh and aerosol
        Tdirs = self.transmittance_dir(aots, self.air_mass, rot=self.rot)
        if not behavior.PREVIOUS_BEHAVIOR:
            Tdirs = Tdirs.where(valid_aot)
        Tdirs.name = 'Tdir'
        Tdirs.attrs['description'] = 'direct transmittance due to rayleigh and aerosol for total air mass'

        # merge into dataset
        self.atmo_img = xr.merge([aots, Rdiffs, Tdirs])
        self.atmo_img.attrs['description'] = "atmospheric parameters for rayleigh and aerosol components",
        self.atmo_img.attrs['aerosol_model'] = self.aerosol_model



# Backward-compatible re-exports. The implementation lives in hgrs.legacy.spectral.
from .legacy.spectral import (
    Gamma2sigma,
    Spectral,
    gaussian,
    super_gaussian,
    super_gaussian_fwhm2sigma,
)
