r'''
Atmospheric and sunglint correction kernel of hGRS: gaseous absorption, water vapor, aerosol and
sunglint retrievals, spectral convolution.

Physical model
==============

Notation
--------

.. list-table::
   :widths: 25 75

   * - :math:`\lambda`
     - central wavelength of the band (``wl``, nm)
   * - :math:`\theta_s,\ \theta_v,\ \phi`
     - solar zenith, viewing zenith and relative azimuth angles (``sza``, ``vza``, ``raa``); the LUTs are
       read with :math:`\phi_{LUT} = (180^\circ - \phi) \bmod 360^\circ`
   * - :math:`M`
     - two-way geometric air mass, :math:`M = 1/\cos\theta_s + 1/\cos\theta_v` (``air_mass``)
   * - :math:`R_{TOA}(\lambda)`
     - top-of-atmosphere reflectance (``Rtoa``)
   * - :math:`T_g(\lambda),\ T_{wv}(\lambda)`
     - transmittances of the absorbing gases other than water vapor and of water vapor
   * - :math:`W`
     - total column water vapor (``tcwv``, kg m\ :sup:`-2`)
   * - :math:`\tau_{ref}`
     - aerosol optical thickness (AOT) at the reference wavelength 550 nm (``aot_ref``)
   * - :math:`\tau_a(\lambda),\ \tau_R(\lambda)`
     - aerosol and Rayleigh optical thicknesses (``aot_lut``, ``rot``)
   * - :math:`P,\ P_0`
     - surface pressure (CAMS) and reference pressure of the Rayleigh optical thickness,
       :math:`P_0` = 1013.25 hPa
   * - :math:`\varepsilon(\lambda)`
     - mean spectral shape of the sunglint reflectance at small angles (``sunglint_eps``)
   * - :math:`B`
     - sunglint amplitude, i.e. the sunglint reflectance normalized by its spectral shape
       (``brdfg``)
   * - :math:`\Lambda_{wv},\ \Lambda_g,\ \Lambda_a,\ \Lambda_+`
     - wavelength sets used for water vapor (800-1300 nm, ``wl_water_vapor``), sunglint
       (2150-2250 nm, ``wl_sunglint``), aerosol fit (``wl_atmo``) and non-negativity constraint
       (``wl_non_neg``)

Spectral convolution
--------------------

Spectrally resolved quantities :math:`X(\lambda)` (solar irradiance, gaseous transmittance) are
integrated over the spectral response function :math:`S_i` of each band :math:`i`
(:py:meth:`Spectral.convolve`):

.. math::

   X_i = \frac{\int X(\lambda)\, S_i(\lambda)\, d\lambda}{\int S_i(\lambda)\, d\lambda},
   \qquad
   S_i(\lambda) = \exp\left(-\frac{|\lambda - \lambda_i|^p}{2\sigma_i^p}\right)

with :math:`p = 2` (Gaussian) or :math:`p \neq 2` (super-Gaussian, :py:meth:`Spectral.convolve2`), and
:math:`\sigma_i` derived from the full width at half maximum :math:`\Gamma_i` of the band
(see :py:func:`Gamma2sigma` and :py:func:`super_gaussian_fwhm2sigma`).

Gaseous absorption
------------------

The optical thickness of the absorbing gases other than water vapor is built from normalized
absorption spectra :math:`\tau^*` scaled by the CAMS columns :math:`c` and the pressure
(:py:meth:`Algo.get_gaseous_optical_thickness`):

.. math::

   \tau_g(\lambda) = c_{O_3}\tau^*_{O_3} + c_{CH_4}\tau^*_{CH_4} + c_{NO_2}\tau^*_{NO_2}
   + \frac{P}{1000}\left[\tau^*_{CO} + \kappa\left(\tau^*_{CO_2} + \tau^*_{O_2} + \tau^*_{O_4}\right)\right]

with :math:`\kappa` = ``coef_abs_scat`` = 0.35, and the transmittance

.. math::

   T_g(\lambda) = \exp\left(-\bar M\, \tau_g(\lambda)\right)

is convolved with the band responses (:math:`\bar M` is the air mass averaged over the image).

Water vapor
-----------

In each pixel of the coarse raster, the TOA reflectance corrected for :math:`T_g` is fitted over
:math:`\Lambda_{wv}` by a linear background attenuated by water vapor (:py:class:`WaterVapor`):

.. math::

   R(\lambda) \simeq T_{wv}(\lambda; W, \bar M)\, (a\lambda + b), \qquad \lambda\ \text{in µm}

where :math:`T_{wv}` is read in a pre-computed LUT. :math:`(W, a, b)` are obtained by non-linear least
squares within the bounds :math:`[0, 60] \times [-10, 1] \times [0, 1]`, and the signal is then divided
by :math:`T_{wv}(\lambda; W)`.

Atmospheric signal
------------------

The atmospheric (Rayleigh + aerosol) diffuse reflectance is read from the radiative transfer LUT of the
selected OPAC aerosol model:

.. math::

   R_{diff}(\lambda; \tau_{ref}) = \frac{I(\lambda; \tau_{ref}, \theta_s, \theta_v, \phi_{LUT})}{\cos\theta_s}

and the direct (sun-surface-sensor) transmittance is

.. math::

   T_{dir}(\lambda; \tau_{ref}) = \exp\left[-\left(\tau_R(\lambda)\frac{P}{P_0} + \tau_a(\lambda; \tau_{ref})\right) M\right]

Sunglint
--------

The sunglint is described by a spectrally flat bidirectional reflectance attenuated by the direct
transmittance, with the spectral shape :math:`\varepsilon(\lambda)` of the Fresnel reflection:

.. math::

   R_g(\lambda) = B\, \frac{T_{dir}(\lambda)\, \varepsilon(\lambda)}
   {\left\langle T_{dir}\, \varepsilon \right\rangle_{\Lambda_g}}

where :math:`\langle\cdot\rangle_{\Lambda_g}` is the average over the SWIR bands of :math:`\Lambda_g`, so
that :math:`B` is the sunglint reflectance in these bands. At full resolution, :math:`B` is computed
from the SWIR bands, where the water-leaving signal is negligible:

.. math::

   B = \left\langle \frac{R_{corr}(\lambda)}{T_{dir}(\lambda)\, \varepsilon(\lambda)} \right\rangle_{\Lambda_g}

Aerosol retrieval
-----------------

The TOA signal of a water pixel is modeled as (:py:meth:`Aerosol.toa_simu`)

.. math::

   R_{sim}(\lambda; \tau_{ref}, B) = R_{diff}(\lambda; \tau_{ref}) + R_g(\lambda; \tau_{ref}, B)

and :math:`(\tau_{ref}, B)` are retrieved on the coarse raster by constrained optimization
(SLSQP, :py:meth:`Aerosol.solve`):

.. math::

   (\hat\tau_{ref}, \hat B) = \underset{\tau_{ref},\, B}{\operatorname{argmin}}
   \sum_{\lambda \in \Lambda_a} \left[R(\lambda) - R_{sim}(\lambda; \tau_{ref}, B)\right]^2
   \quad \text{subject to} \quad
   \min_{\lambda \in \Lambda_+} \left[R(\lambda) - R_{sim}(\lambda; \tau_{ref}, B)\right] \geq 0

that is, the water-leaving reflectance must stay non-negative in the visible and NIR bands of
:math:`\Lambda_+`. :math:`\tau_{ref}` is bounded by the CAMS values (see
:py:meth:`hgrs.hgrs_process.Process.execute`) and :math:`B \in [0, 1.3]`.

Remote-sensing reflectance
--------------------------

With the retrieved parameters interpolated at full resolution:

.. math::

   R_{corr}(\lambda) = \frac{R_{TOA}(\lambda)}{T_g(\lambda)\, T_{wv}(\lambda)} - R_{diff}(\lambda; \hat\tau_{ref})

.. math::

   R_{rs}(\lambda) = \frac{R_{corr}(\lambda) - T_{dir}(\lambda)\, \varepsilon(\lambda)\, B}
   {\pi\, T_{\downarrow}(\lambda; \theta_s)\, T_{\downarrow}(\lambda; \theta_v)^{1.05}}

where :math:`T_{\downarrow}` is the total (direct + diffuse) transmittance of the downwelling irradiance
(LUT ``Ttot_Ed``), evaluated for the mean geometry and the mean retrieved :math:`\tau_{ref}`.

Processing sequence
===================

Typical sequence (``l1c`` = L1C raster with ``Rtoa``, ``fwhm`` and angles), as in
:py:meth:`hgrs.hgrs_process.Process.execute`::

    prod = Algo(l1c, xcoarsen=20, ycoarsen=20)
    prod.apply_water_masks()
    prod.get_coarse_masked_raster()            # mega-pixels of water pixels
    prod.get_gaseous_transmittance()
    prod.other_gas_correction()                # R / T_g
    wv = WaterVapor(prod)
    wv.solve()
    prod.get_wv_transmittance_raster(wv.water_vapor)
    prod.water_vapor_correction()              # R / T_wv
    aero = Aerosol(prod, aerosol_model='COAV_rh70')
    aero.solve()
    aero.get_atmo_parameters(prod)             # R_diff, T_dir rasters
'''

import os
from pkg_resources import resource_filename
import importlib_resources
import yaml

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

opj = os.path.join

configfile = importlib_resources.files(__package__).joinpath('config.yml')
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

HGRSDATA = config['path']['data_root']
TOALUT = config['path']['toa_lut']
TRANSLUT = config['path']['trans_lut']

LUT_FILE = opj(HGRSDATA, TOALUT)
TRANS_LUT_FILE = opj(HGRSDATA, TRANSLUT)
ABS_GAS_FILE = resource_filename(__package__, 'data/lut/lut_abs_opt_thickness_normalized.nc')
WATER_VAPOR_TRANSMITTANCE_FILE = resource_filename(__package__, 'data/lut/water_vapor_transmittance.nc')


class Product():
    r'''
    Hyperspectral L1C image prepared for hGRS: spectral and processing parameters, auxiliary data and
    water masks.

    Parameters
    ----------
    l1c_obj : xarray.Dataset
        L1C image with the TOA reflectance ``Rtoa`` (wl, y, x), the band widths ``fwhm`` and the angles
        ``sza``, ``vza``, ``raa`` (see :py:class:`hgrs.driver.Driver`).
    xcoarsen, ycoarsen : int
        Number of pixels along x and y aggregated into one mega-pixel of the coarse raster.
    expon : float
        Exponent of the super-Gaussian spectral response functions.

    Notes
    -----
    Water pixels (:py:meth:`apply_water_masks`) are those satisfying

    .. math::

       NDWI = \frac{R(\lambda_{G}) - R(\lambda_{NIR})}{R(\lambda_{G}) + R(\lambda_{NIR})} > 0.01, \quad
       \frac{R(\lambda_{G}) - R(1600)}{R(\lambda_{G}) + R(1600)} > 0.1, \quad
       R(2200) < 0.11

    with the reflectances averaged over 540-570 nm (:math:`\lambda_G`), 850-882 nm
    (:math:`\lambda_{NIR}`), 1580-1650 nm and 2150-2250 nm. A mega-pixel is processed when it contains at
    least ``pixel_percentage`` = 20 % of water pixels.
    '''

    def __init__(self,
                 l1c_obj=None,
                 xcoarsen=20,
                 ycoarsen=20,
                 expon=2):

        # spectral parameters
        self.wl_water_vapor = slice(800, 1300)
        self.wl_sunglint = slice(2150, 2250)
        # self.wl_atmo = slice(950, 2450)
        self.wl_atmo = [1000, 1050, 1075, 1100, 1200, 1300, 1600, 1650, 1700, 2150, 2200, 2250]
        self.wl_non_neg = [430, 490, 560, 650, 750, 800, 865, 1020]
        self.wl_to_remove = [(935, 967), (1105, 1170), (1320, 1490), (1778, 2033), (2465, 2550)]
        self.wl_green = slice(540, 570)
        self.wl_nir = slice(850, 882)
        self.wl_1600 = slice(1580, 1650)
        self.wl_rgb = [30, 20, 10]

        # image chunking and coarsening parameters
        self.xcoarsen = xcoarsen
        self.ycoarsen = ycoarsen
        self.Npix_per_megapix = self.xcoarsen * self.ycoarsen
        self.block_size = 2
        # minimum percentage of water pixel within the mega-pixel to enable processing
        self.pixel_percentage = 20
        self.pixel_threshold = self.pixel_percentage / 100 * self.Npix_per_megapix

        # number of digits to keep for angle values
        self.ang_resol = 1

        # pre-computed auxiliary data
        self.lut_file = LUT_FILE
        self.trans_lut_file = TRANS_LUT_FILE
        self.abs_gas_file = ABS_GAS_FILE
        self.water_vapor_transmittance_file = WATER_VAPOR_TRANSMITTANCE_FILE

        # mask thresholding parameters
        self.sunglint_threshold = 0.11
        self.ndwi_threshold = 0.01
        self.green_swir_index_threshold = 0.1

        # atmosphere auxiliary data
        # TODO get them from CAMS
        self.pressure = 1010
        self.to3c = 6.5e-3
        self.tno2c = 3e-6
        self.tch4c = 1e-2
        self.psl = 1013
        self.coef_abs_scat = .35

        self.altitude = 0

        # xarray object to be processed
        self.raster = l1c_obj.copy()

        self.fwhm = self.raster.fwhm.reset_coords(drop=True)  # .to_dataframe()
        self.wl = self.raster.wl
        self.sza_mean = np.nanmean(self.raster.sza)
        self.vza_mean = np.nanmean(self.raster.vza)
        self.raa_mean = np.nanmean(self.raster.raa)
        self.get_air_mass()

        self.Tg_other = None

        self.load_auxiliary_data()

        # spectral function for sensor response convolution
        # exponent of the super-gaussian spectral response function
        self.expon = expon
        # set the convolution module
        self.spectral = Spectral(self.wl, self.fwhm.values)

    def load_auxiliary_data(self):

        # ---------------------------------------
        # Load pre-computed radiative transfer LUT
        # ---------------------------------------
        logging.info('Load pre-computed radiative transfer LUT')

        # get LUT
        self.gas_lut = xr.open_dataset(self.abs_gas_file)
        self.aero_lut = xr.open_dataset(self.lut_file).isel(wind=1)
        self.Ttot_Ed = xr.open_dataset(self.trans_lut_file).isel(wind=1)
        self.Twv_lut = xr.open_dataset(self.water_vapor_transmittance_file).interp(wl=self.wl)

        # convert wavelength in nanometer
        self.aero_lut['wl'] = self.aero_lut['wl'] * 1000
        self.aero_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'
        self.Ttot_Ed['wl'] = self.Ttot_Ed['wl'] * 1e3
        self.Ttot_Ed['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'

        # get hgrs auxdata
        self.auxdata = AuxData(self.wl)

    def get_ndwi(self):
        r'''Normalized difference water index,
        :math:`NDWI = (R_{G} - R_{NIR}) / (R_{G} + R_{NIR})`.'''
        green = self.raster.Rtoa.sel(wl=self.wl_green).mean(dim='wl')
        nir = self.raster.Rtoa.sel(wl=self.wl_nir).mean(dim='wl')
        self.ndwi = (green - nir) / (green + nir)

    def get_green_swir_index(self):
        r'''Green-SWIR normalized index, :math:`(R_{G} - R_{1600}) / (R_{G} + R_{1600})`.'''
        green = self.raster.Rtoa.sel(wl=self.wl_green).mean(dim='wl')
        b1600 = self.raster.Rtoa.sel(wl=self.wl_1600).mean(dim='wl')
        self.green_swir_index = (green - b1600) / (green + b1600)

    def get_b2200(self):
        '''Mean TOA reflectance over the sunglint bands (2150-2250 nm).'''
        self.b2200 = self.raster.Rtoa.sel(wl=self.wl_sunglint).mean(dim='wl')

    def apply_water_masks(self):
        '''Set ``Rtoa`` to NaN outside the water pixels (see the class notes for the thresholds).'''
        self.get_ndwi()
        self.get_green_swir_index()
        self.get_b2200()
        self.raster['Rtoa'] = self.raster.Rtoa.where(self.ndwi > self.ndwi_threshold). \
            where(self.b2200 < self.sunglint_threshold). \
            where(self.green_swir_index > self.green_swir_index_threshold).load()

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
        '''Round the angles to ``ang_resol`` digits (reduces the number of LUT interpolations).'''
        for param in ['sza', 'vza', 'raa']:
            self.raster[param] = self.raster[param].round(self.ang_resol)

    def get_air_mass(self, raster_name='raster', round=True, digit_resol=3):
        r'''
        Two-way geometric air mass

        .. math::

           M = \frac{1}{\cos\theta_s} + \frac{1}{\cos\theta_v}

        stored as ``air_mass`` in the raster; its mean over the image is ``air_mass_mean``.
        '''
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

    @staticmethod
    def Gamma2sigma(Gamma):
        '''Function to convert FWHM (Gamma) to standard deviation (sigma), see :py:func:`Gamma2sigma`'''
        return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)

    @staticmethod
    def gaussian(x, mu, sigma):
        '''Normalized Gaussian distribution, see :py:func:`gaussian`'''
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    @njit(fastmath=True)
    def super_gaussian(x,
                       amplitude=1.0,
                       mu=0.0,
                       sigma=1.0,
                       expon=10.0):
        '''
        Super-Gaussian distribution, see :py:func:`super_gaussian`
        '''

        sigma = max(1.e-15, sigma)
        return amplitude / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-np.abs(x - mu) ** expon / (2 * sigma ** expon))

    @staticmethod
    @njit(fastmath=True)
    def super_gaussian_fwhm2sigma(fwhm,
                                  expon):
        '''
        Function to convert FWHM to standard deviation (sigma) of the super-gaussian distribution,
        see :py:func:`super_gaussian_fwhm2sigma`
        '''
        return fwhm / 2 * (2 * np.log(2)) ** (-1 / expon)

    def plot_rsr(self):

        wl_ref = np.linspace(360, 2550, 10000)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for mu, fwhm in self.fwhm.iterrows():
            sig = self.Gamma2sigma(fwhm.values)
            rsr = self.gaussian(wl_ref, mu, sig)
            axs.plot(wl_ref, rsr, '-k', lw=0.5, alpha=0.4)
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Spectral response function')

        return fig

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
    '''
    :py:class:`Product` with the gaseous corrections and the coarse (mega-pixel) rasters used for the
    retrievals.
    '''

    def __init__(self, l1c_obj=None, xcoarsen=20, ycoarsen=20, expon=2):
        Product.__init__(self, l1c_obj, xcoarsen, ycoarsen, expon)

    def get_pressure(self, alt, psl):
        r'''Compute the pressure for a given altitude (barometric formula)

        .. math::

           P(z) = P_{sl} \left(1 - \frac{0.0065\, z}{288.15}\right)^{5.255}

        :param alt: altitude :math:`z` in meters (float or np.array)
        :param psl: pressure at sea level :math:`P_{sl}` in hPa
        :return: pressure at the given altitude in hPa'''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    def get_coarse_raster(self, variables=['sza', 'vza', 'raa', 'air_mass', 'Rtoa']):
        '''Average ``variables`` over mega-pixels of ``xcoarsen`` x ``ycoarsen`` pixels.'''
        self.coarse_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen, boundary="pad").mean()

    def get_coarse_masked_raster(self, variables=['sza', 'vza', 'raa', 'air_mass', 'Rtoa']):
        '''
        Average ``variables`` over mega-pixels of ``xcoarsen`` x ``ycoarsen`` pixels (masked pixels
        are ignored) and count the valid water pixels of each mega-pixel (``water_pixel_number``),
        used to select and weight the mega-pixels in the retrievals.
        '''
        self.coarse_masked_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen,
                                                                   boundary="pad").mean()
        self.coarse_masked_raster['water_pixel_number'] = self.raster['Rtoa']. \
            isel(wl=slice(10, 20)).mean(dim='wl'). \
            coarsen(x=self.xcoarsen, y=self.ycoarsen, boundary="pad").count()

    def get_gaseous_optical_thickness(self):
        r'''
        Optical thickness of the absorbing gases other than water vapor

        .. math::

           \tau_g(\lambda) = c_{O_3}\tau^*_{O_3} + c_{CH_4}\tau^*_{CH_4} + c_{NO_2}\tau^*_{NO_2}
           + \frac{P}{1000}\left[\tau^*_{CO} + \kappa\left(\tau^*_{CO_2} + \tau^*_{O_2} + \tau^*_{O_4}\right)\right]

        where :math:`\tau^*` are the normalized optical thicknesses of the LUT ``abs_gas_file``,
        :math:`c` the columns ``to3c``, ``tch4c``, ``tno2c``, :math:`P` the pressure (hPa) and
        :math:`\kappa` = ``coef_abs_scat``. Result in ``abs_gas_opt_thick``.
        '''
        gas_lut = self.gas_lut

        ot_o3 = gas_lut.o3 * self.to3c
        ot_ch4 = gas_lut.ch4 * self.tch4c
        ot_no2 = gas_lut.no2 * self.tno2c
        ot_air = (gas_lut.co + self.coef_abs_scat * gas_lut.co2 +
                  self.coef_abs_scat * gas_lut.o2 +
                  self.coef_abs_scat * gas_lut.o4) * self.pressure / 1000
        self.abs_gas_opt_thick = ot_ch4 + ot_no2 + ot_o3 + ot_air

    def get_gaseous_transmittance(self):
        r'''
        Transmittance of the absorbing gases other than water vapor convolved with the band responses

        .. math::

           T_{g,i} = \frac{\int \exp\left(-\bar M \tau_g(\lambda)\right) S_i(\lambda)\, d\lambda}{\int S_i(\lambda)\, d\lambda}

        with :math:`\bar M` = ``air_mass_mean``. Result in ``Tg_other``.
        '''

        self.get_gaseous_optical_thickness()
        wl_ref = self.gas_lut.wl  # .values
        Tg = np.exp(- self.air_mass_mean * self.abs_gas_opt_thick)

        # TODO check which one better
        # self.Tg_other = self.spectral.convolve2(Tg,name='Ttot',expon=self.expon)
        self.Tg_other = self.spectral.convolve(Tg, name='Ttot')

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
        '''Divide ``variable`` of the raster ``raster_name`` by ``Tg_other`` (done only once, flagged
        by the attribute ``other_gas_correction``).'''
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
        Divide ``variable`` of the raster ``raster_name`` by the water vapor transmittance
        ``Twv_raster`` (see :py:meth:`get_wv_transmittance_raster`).

        :param raster_name: name of the raster attribute to correct
        :param variable: variable to correct
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
        r'''
        Raster of the water vapor transmittance :math:`T_{wv}(\lambda; W, \bar M)` from the retrieved
        water vapor ``tcwv`` (rounded to 0.1 kg m-2) and the LUT ``water_vapor_transmittance_file``.
        Result in ``Twv_raster``.

        :param tcwv_raster: xarray.Dataset with ``tcwv`` (see :py:class:`WaterVapor`)
        '''
        tcwv_vals = tcwv_raster.tcwv.round(1)
        tcwvs = np.unique(tcwv_vals)
        tcwvs = tcwvs[~np.isnan(tcwvs)]
        # TODO improve for air_mass raster
        Twvs = self.Twv_lut.Twv.interp(air_mass=self.air_mass_mean).interp(tcwv=tcwvs, method='linear').drop('air_mass')
        self.Twv_raster = Twvs.interp(tcwv=tcwv_vals, method='nearest')

    def get_full_resolution(self, xarr):
        '''Interpolate ``xarr`` (coarse raster) onto the full resolution grid.'''
        return xarr.interp(x=self.raster.x, y=self.raster.y)


class Solver():
    '''Common tools of the pixel-wise retrievals (uncertainties, smoothing).'''

    def __init__(self):
        pass

    def errFit(self, hess_inv, resVariance):
        r'''
        Error/uncertainty of the estimated parameters

        .. math::

           \sigma_{x_k} = \sqrt{\left[(J^T J)^{-1}\right]_{kk}\, s^2},
           \qquad s^2 = \frac{\sum_j r_j^2}{n - p}

        with :math:`J` the Jacobian of the residuals :math:`r` at the solution, :math:`n` the number of
        wavelengths and :math:`p` the number of parameters.

        :param hess_inv: inverse of the approximated Hessian :math:`(J^T J)^{-1}`
        :param resVariance: variance of the residuals :math:`s^2`
        :return: standard deviations of the parameters
        '''
        return np.sqrt(np.diag(hess_inv * resVariance))

    def conv_mapping(self, x):
        """
        Nan-mean convolution: mean of the non-NaN values of the window; if the central value is NaN,
        it is replaced by the mean of its neighbors (gap filling).
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
        r'''
        Function to convolve parameter image with uncertainty image (weighted moving average)

        .. math::

           \tilde x_{ij} = \frac{\sum_{(k,l) \in W_{ij}} w_{kl}\, x_{kl}}{\sum_{(k,l) \in W_{ij}} w_{kl}}

        computed where :math:`w_{ij} > 0`, :math:`W_{ij}` being the window centered on :math:`(i,j)`.

        :param image: parameter image
        :param weight: weight image (e.g., number of water pixels or inverse variance)
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
    r'''
    Retrieval of the total column water vapor :math:`W` on the coarse raster.

    In each mega-pixel, the reflectance over ``prod.wl_water_vapor`` (800-1300 nm) is fitted with

    .. math::

       R(\lambda) \simeq T_{wv}(\lambda; W, \bar M)\, (a\lambda + b)

    by non-linear least squares (:py:func:`scipy.optimize.least_squares`), with
    :math:`(W, a, b) \in [0, 60] \times [-10, 1] \times [0, 1]` (:math:`\lambda` in µm). The
    uncertainty ``tcwv_std`` is given by :py:meth:`Solver.errFit`.

    :param prod: :py:class:`Algo` object, corrected for the other gases
    :param raster_name: name of the raster attribute of ``prod`` to process
    :param variable: reflectance variable
    '''

    def __init__(self, prod,
                 raster_name='coarse_masked_raster',
                 variable='Rtoa'):

        self.prod = prod
        self.raster = prod.__dict__[raster_name]
        self.air_mass = prod.air_mass_mean
        # get data for the subset of "water vapor" wavelengths
        data = self.raster[variable].sel(wl=prod.wl_water_vapor)
        self.data = data
        self.nwl, self.height, self.width = data.shape
        self.x = data.x
        self.y = data.y
        self.wl = data.wl

        # TODO improve to process the air mass raster instead of scalar mean value
        # TODO check impact of method = 'nearest'
        self.Twv_ = prod.Twv_lut.Twv.interp(wl=self.data.wl).interp(air_mass=self.air_mass)
        self.Twv_['wl'] = self.Twv_['wl'] / 1000
        self.wl_mic = self.Twv_.wl.values

    def toa_simu(self, wl, Twv, tcwv, a, b):
        r'''Simulated reflectance :math:`T_{wv}(\lambda; W)\,(a\lambda + b)`, ``wl`` in micron
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

    def solve(self, x0=[2, -0.04, 0.1]):
        '''
        Run the fit on every mega-pixel with enough water pixels (parallel processing of blocks of
        ``block_size`` mega-pixels). Result in ``water_vapor`` (``tcwv``, ``tcwv_std``).

        :param x0: first guess of :math:`(W, a, b)`
        '''

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


class Aerosol(Solver):
    r'''
    Retrieval of the aerosol optical thickness :math:`\tau_{ref}` (550 nm) and of the sunglint
    amplitude :math:`B` on the coarse raster, under the constraint of non-negative water reflectance.

    :param prod: :py:class:`Algo` object corrected for gaseous absorption (other gases and water vapor)
    :param aerosol_model: OPAC aerosol model of the LUT (e.g., ``'COAV_rh70'``)
    :param first_guess: first guess of :math:`(\tau_{ref}, B)`
    :param aot550_limits: bounds of :math:`\tau_{ref}` for the optimization
    :param raster_name: name of the raster attribute of ``prod`` to process
    :param variable: reflectance variable
    '''

    def __init__(self, prod,
                 aerosol_model='COAV_rh70',
                 first_guess=[0.01, 0],
                 aot550_limits=[0.002, 0.8],
                 raster_name='coarse_masked_raster',
                 variable='Rtoa'):

        self.prod = prod
        self.aerosol_model = aerosol_model
        self.raster = prod.__dict__[raster_name]
        self.auxdata = prod.auxdata
        self.aero_lut = prod.aero_lut

        # set box limits in aod550 for non-linear optimization
        self.aod550_min = aot550_limits[0]
        self.aod550_max = aot550_limits[1]
        self.first_guess = first_guess

        # get full resolution parameters
        self.xfull = prod.raster.x
        self.yfull = prod.raster.y

        # get data for the subset of "black water" wavelengths
        self.data = self.raster[variable]
        data = self.data
        self.wl_atmo = prod.wl_atmo
        self.wl_non_neg = prod.wl_non_neg
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
        self.wl_sunglint = self.prod.wl_sunglint

        self.prepare_lut(self.wl)

    def prepare_lut(self, wl):
        r'''
        Reduce the LUTs to the image geometry (mean angles), the aerosol model and the wavelengths
        ``wl``, on a grid of 101 values of :math:`\tau_{ref}` (0 and 100 log-spaced values in
        [0.001, 0.8]):

        - ``Rtoa_lut``: :math:`R_{diff}(\lambda; \tau_{ref}) = I(\lambda; \tau_{ref}, \theta_s, \theta_v, \phi_{LUT}) / \cos\theta_s`
        - ``aot_lut``: :math:`\tau_a(\lambda; \tau_{ref})`
        - ``rot``: :math:`\tau_R(\lambda)\, P / P_0`
        - ``sunglint_eps``: :math:`\varepsilon(\lambda)`

        :param wl: wavelengths (nm)
        '''
        auxdata = self.auxdata
        sza = self.sza
        vza = self.vza
        raa_lut = self.raa_lut

        self.sunglint_eps = auxdata.sunglint_eps.interp(wl=wl)
        self.rot = auxdata.rot.interp(wl=wl) * self.pressure / self.auxdata.pressure_rot_ref

        aot_refs = [0, *np.logspace(-3, np.log10(0.8), 100)]
        self.aot_lut = self.aero_lut.sel(model=self.aerosol_model).aot.interp(wl=wl, method='quadratic').interp(
            aot_ref=aot_refs,
            method='quadratic').dropna('aot_ref')  # sel(wl=wl_glint)

        norm_radiance = self.aero_lut.sel(model=self.aerosol_model
                                          ).I.interp(vza=vza, azi=raa_lut, method='linear'
                                                     ).interp(sza=sza, method='quadratic').squeeze()
        self.Rtoa_lut = norm_radiance.interp(wl=wl, method='quadratic') \
                            .interp(aot_ref=aot_refs, method='quadratic').dropna('aot_ref') / np.cos(np.radians(sza))

    def transmittance_dir(self, aot, M, rot=0):
        r'''Direct transmittance :math:`T_{dir} = \exp\left[-(\tau_R + \tau_a) M\right]`'''
        return np.exp(-(rot + aot) * M)

    def toa_simu(self, aot, rot, Rtoa_lut, sunglint_eps, aot_ref, BRDFg):
        r'''
        Simulated TOA reflectance of a black water body

        .. math::

           R_{sim}(\lambda) = R_{diff}(\lambda; \tau_{ref})
           + B\, \frac{T_{dir}(\lambda)\, \varepsilon(\lambda)}{\left\langle T_{dir}\, \varepsilon \right\rangle_{\Lambda_g}}

        :param aot: spectral AOT LUT :math:`\tau_a(\lambda; \tau_{ref})`
        :param rot: Rayleigh optical thickness :math:`\tau_R(\lambda) P/P_0`
        :param Rtoa_lut: LUT of :math:`R_{diff}(\lambda; \tau_{ref})`
        :param sunglint_eps: spectral shape of the sunglint :math:`\varepsilon(\lambda)`
        :param aot_ref: :math:`\tau_{ref}`
        :param BRDFg: sunglint amplitude :math:`B`
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
        r'''Residuals :math:`R(\lambda) - R_{sim}(\lambda; \tau_{ref}, B)`'''
        return (y - self.toa_simu(aot, rot, Rtoa_lut, sunglint_eps, *x))  # /sigma

    def cost_func(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        r'''Cost function :math:`\sum_\lambda \left[R(\lambda) - R_{sim}(\lambda)\right]^2`'''
        return np.sum((self.func(x, aot, rot, Rtoa_lut, sunglint_eps, y) ** 2))

    def constraint(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        r'''Non-negativity constraint :math:`\min_\lambda \left[R(\lambda) - R_{sim}(\lambda)\right] \geq 0`'''
        return np.min((self.func(x, aot, rot, Rtoa_lut, sunglint_eps, y)))

    def solve(self, x0=[0.005, 0.]):
        r'''
        Retrieve :math:`(\tau_{ref}, B)` on every mega-pixel with enough water pixels

        .. math::

           (\hat\tau_{ref}, \hat B) = \underset{\tau_{ref},\, B}{\operatorname{argmin}}
           \sum_{\lambda \in \Lambda_a} \left[R(\lambda) - R_{sim}(\lambda)\right]^2
           \quad \text{s.t.} \quad
           \min_{\lambda \in \Lambda_+} \left[R(\lambda) - R_{sim}(\lambda)\right] \geq 0

        with :py:func:`scipy.optimize.minimize` (SLSQP, 10 iterations at most),
        :math:`\tau_{ref}` within ``aot550_limits`` and :math:`B \in [0, 1.3]`. :math:`\Lambda_a` and
        :math:`\Lambda_+` are ``prod.wl_atmo`` and ``prod.wl_non_neg``. The solution of a successful
        optimization is the first guess of the next mega-pixel of the block.

        Result in ``aero_img``: ``aot_ref``, ``brdfg``, and, as quality indicators, the final cost
        (``aot_ref_std``) and the squared norm of its gradient (``brdfg_std``).

        :param x0: not used, the first guess is ``first_guess``
        '''

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
                                       constraints=cons, options={'maxiter': 10}
                                       )
                    xres = min_res.x
                    if min_res.success:
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

        self.aero_img = xr.Dataset(dict(aot_ref=(["y", "x"], result[:, :, 0].T),
                                        brdfg=(["y", "x"], result[:, :, 1].T),
                                        aot_ref_std=(["y", "x"], result[:, :, 2].T),
                                        brdfg_std=(["y", "x"], result[:, :, 3].T)
                                        ),
                                   coords=dict(
                                       x=self.x,
                                       y=self.y),
                                   attrs=dict(
                                       description="aerosol and sunglint retrieval from coarse resolution data",
                                       aerosol_model=self.aerosol_model)
                                   )

    def smoothing(self,
                  weights,
                  windows=np.array([1, 1]),
                  mask=np.ones((3, 3))
                  ):
        '''
        Smooth and gap-fill the retrieved ``aot_ref``: weighted moving average (:py:meth:`Solver.filter2d`)
        followed by a NaN-mean filter over the footprint ``mask`` (:py:meth:`Solver.conv_mapping`).
        Result in ``aero_img['aot_ref_smoothed']``.

        :param weights: weight image (e.g., number of water pixels per mega-pixel)
        :param windows: window of the weighted average
        :param mask: footprint of the NaN-mean filter
        '''

        #weights = (1 / self.aero_img['aot_ref_std'] ** 2).__deepcopy__().to_numpy().astype(float)
        param = self.aero_img['aot_ref'].__deepcopy__().to_numpy().astype(float)
        aot_ref_smoothed = self.filter2d(param, weights, windows)
        res = ndimage.generic_filter(aot_ref_smoothed, function=self.conv_mapping, footprint=mask, mode='nearest')

        self.aero_img['aot_ref_smoothed'] = xr.DataArray(res, coords=dict(y=self.aero_img.y, x=self.aero_img.x))

    def get_aot_full_resolution(self):
        '''Linear interpolation (and extrapolation) of ``aot_ref_smoothed`` at full resolution.'''
        # TODO change fill_value (extrapolate is not safe) with median of retrieval, for instance
        self.aot_ref_full = \
            self.aero_img['aot_ref_smoothed'].interp(x=self.xfull, y=self.yfull,
                                                     method='linear', kwargs={"fill_value": "extrapolate"})

    def get_atmo_parameters(self,
                            prod):
        r'''
        Rasters of the atmospheric parameters for the smoothed :math:`\tau_{ref}` (rounded to 0.001):
        spectral AOT ``aot``, diffuse reflectance ``Rtoa_diff`` and direct transmittance ``Tdir``.
        Result in ``atmo_img``.

        :param prod: :py:class:`Algo` object
        '''

        wl = prod.coarse_masked_raster.wl
        weights = prod.coarse_masked_raster['water_pixel_number'].__deepcopy__().to_numpy().astype(float)

        # get LUT for desired wavelengths
        self.prepare_lut(wl)
        self.smoothing(weights)
        self.get_aot_full_resolution()

        # construct aot raster
        aot_ref_median = self.aero_img.aot_ref_smoothed.median()
        aot_ref_vals = self.aero_img['aot_ref_smoothed'].fillna(aot_ref_median).round(3)
        aot_refs = np.unique(aot_ref_vals)
        aot_refs = aot_refs[~np.isnan(aot_refs)]
        # TODO update LUT for aot< 0.001
        aot_refs[aot_refs < 0.002] = 0.002

        # if rounded aot_ref has unique value
        if len(aot_refs) == 1:
            aot_refs = np.concatenate([aot_refs, 1.2 * aot_refs])
        aots = self.aot_lut.interp(aot_ref=aot_refs, method='linear')
        aots = aots.interp(aot_ref=aot_ref_vals, method='nearest')

        aots.name = 'aot'
        aots.attrs['description'] = 'spectral aerosol optical thickness'

        # construct raster for diffuse atmospheric reflectance
        Rdiffs = self.Rtoa_lut.interp(aot_ref=aot_refs, method='linear')
        Rdiffs = Rdiffs.interp(aot_ref=aot_ref_vals, method='nearest')
        Rdiffs.name = 'Rtoa_diff'
        Rdiffs.attrs['description'] = 'top-of-atmosphere atmosphere reflectance'

        # construct raster for direct transmittance due to rayleigh and aerosol
        Tdirs = self.transmittance_dir(aots, self.air_mass, rot=self.rot)
        Tdirs.name = 'Tdir'
        Tdirs.attrs['description'] = 'direct transmittance due to rayleigh and aerosol for total air mass'

        # merge into dataset
        self.atmo_img = xr.merge([aots, Rdiffs, Tdirs])
        self.atmo_img.attrs['description'] = "atmospheric parameters for rayleigh and aerosol components",
        self.atmo_img.attrs['aerosol_model'] = self.aerosol_model


@njit(fastmath=True)
def Gamma2sigma(Gamma):
    r'''Function to convert FWHM (Gamma) to standard deviation (sigma)

    .. math::

       \sigma = \frac{\sqrt{2}\, \Gamma}{2 \sqrt{2 \ln 2}} = \frac{\Gamma}{2\sqrt{\ln 2}}
    '''
    return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)


@njit(fastmath=True)
def gaussian(x, mu, sigma):
    r'''Normalized Gaussian distribution

    .. math::

       g(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
    '''
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


@njit(fastmath=True)
def super_gaussian(x,
                   amplitude=1.0,
                   mu=0.0,
                   sigma=1.0,
                   expon=10.0):
    r'''
    Super-Gaussian distribution

    .. math::

       g(x) = \frac{A}{\sigma\sqrt{2\pi}} \exp\left(-\frac{|x - \mu|^p}{2\sigma^p}\right)

    :param x: abscissa (e.g., wavelength)
    :param amplitude: :math:`A`
    :param mu: center :math:`\mu`
    :param sigma: width :math:`\sigma`
    :param expon: exponent :math:`p` (:math:`p = 2` for a Gaussian)
    :return: :math:`g(x)`
    '''

    sigma = max(1.e-15, sigma)
    return amplitude / (np.sqrt(2 * np.pi) * sigma) * \
        np.exp(-np.abs(x - mu) ** expon / (2 * sigma ** expon))


@njit(fastmath=True)
def super_gaussian_fwhm2sigma(fwhm,
                              expon):
    r'''
    Function to convert FWHM to standard deviation (sigma) of the super-gaussian distribution

    .. math::

       \sigma = \frac{\Gamma}{2} \left(2 \ln 2\right)^{-1/p}

    :param fwhm: full width at half maximum :math:`\Gamma`
    :param expon: exponent :math:`p` of the super-Gaussian
    :return: :math:`\sigma`
    '''
    return fwhm / 2 * (2 * np.log(2)) ** (-1 / expon)


class Spectral():
    r'''
    Convolution of spectral signals with the spectral response functions (SRF) of the sensor bands

    .. math::

       X_i = \frac{\int X(\lambda)\, S_i(\lambda)\, d\lambda}{\int S_i(\lambda)\, d\lambda}

    where :math:`S_i` is a Gaussian (:py:meth:`convolve`) or a super-Gaussian (:py:meth:`convolve2`)
    centered on :math:`\lambda_i` with the full width at half maximum of the band.

    :param central_wl: numpy array of the central wavelengths
    :param fwhm: scalar or numpy array containing full width at half maximum in nm
    '''

    def __init__(self,
                 central_wl,
                 fwhm):
        self.central_wl = central_wl
        if not isinstance(fwhm, np.ndarray):
            fwhm = np.array([fwhm] * len(central_wl))
        fwhm = xr.DataArray(fwhm, name='fwhm',
                            coords={'wl': central_wl},
                            attrs={
                                'definition': 'full width at half maximum of spectral responses modeled as gaussian distributions'})
        self.fwhm = fwhm

    def plot_rsr(self):

        wl_ref = np.linspace(360, 2550, 10000)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for mu, fwhm in self.fwhm.groupby('wl'):
            sig = self.Gamma2sigma(fwhm.values)
            rsr = self.gaussian(wl_ref, mu, sig)
            axs.plot(wl_ref, rsr, '-k', lw=0.5, alpha=0.4)
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Spectral response function')

        return fig

    @staticmethod
    @njit(parallel=True)
    def convolve_(
            wl_signal,
            signal,
            wl,
            fwhm,
    ):
        '''
        Convolution assuming Dirac for signal source spectral response
        :paral wl_signal: wavelength array of spectral signal
        :param signal: numpy of signal to convolve, coord=wl_signal
        :param wl: numpy of wavelength coordinates of signal
        :param fwhm: numpy with data=fwhm containing full width at half maximum in nm
        :return: numpy of convoluted signal
        '''
        Nwl = len(wl)
        signal_ = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = Gamma2sigma(fwhm[ii])
            rsr = gaussian(wl_signal, wl[ii], sig)
            signal_[ii] = np.trapezoid((signal * rsr), wl_signal) / np.trapezoid(rsr, wl_signal)
        return signal_

    @staticmethod
    @njit(parallel=True)
    def convolve2_(
            wl_signal,
            signal,
            wl,
            fwhm,
            expon=2.,
            threshold=1e-6
    ):
        '''
        Convolution assuming Dirac for signal source spectral response
        :paral wl_signal: wavelength array of spectral signal
        :param signal: numpy of signal to convolve, coord=wl_signal
        :param wl: numpy of wavelength coordinates of signal
        :param fwhm: numpy with data=fwhm containing full width at half maximum in nm
        :param threshold: minimum values of the response function to be included in the convolution
        :return: numpy of convoluted signal
        '''

        Nwl = len(wl)
        response = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = super_gaussian_fwhm2sigma(fwhm[ii], expon)
            rsr = super_gaussian(wl_signal, mu=wl[ii], sigma=sig, expon=expon)

            # remove values above a given threshold to speed up computation
            idx = rsr > threshold
            wl_signal_ = wl_signal[idx]
            signal_ = signal[idx]
            rsr = rsr[idx]

            response[ii] = np.trapezoid((signal_ * rsr), wl_signal_) / np.trapezoid(rsr, wl_signal_)
        return response

    def convolve2(self,
                  signal,
                  name='signal',
                  expon=3,
                  threshold=1e-4,
                  info={}):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param signal: xarray spectral signal to convolve, coord=wl
        :param fwhm: xarray with data=fwhm containing full width at half maximum in nm, and coords=wl
        :param info: optional parameter to feed the attributes of the output xarray
        :param threshold: minimum values of the response function to be included in the convolution
        :return:
        '''

        wl_ref = signal.wl.values
        fwhm = self.fwhm.values
        wl = self.fwhm.wl.values
        xdims = signal.dims
        attrs = signal.attrs
        name = signal.name
        if len(xdims) == 1:
            signal_int = self.convolve2_(wl_ref, signal.values, wl, fwhm, expon, threshold=threshold)
            signal_int = xr.DataArray(signal_int, name=name,
                                      coords={'wl': self.fwhm.wl.values},
                                      attrs=attrs)

        else:
            # to handle multidimensional xarray
            xdims = np.array(xdims)
            xdims = xdims[xdims != 'wl']

            xsignal_int = []
            for dim in xdims:
                xsignal_int_ = []
                for value, signal_ in signal.groupby(dim):
                    # print(dim, value)
                    signal_ = signal_.squeeze()
                    _ = self.convolve2_(signal_.wl.values, signal_.values, wl, fwhm, expon)
                    _ = xr.Dataset({name: (['wl'], _)},
                                   coords={'wl': wl,
                                           dim: value})
                    xsignal_int_.append(_)
                xsignal_int.append(xr.concat(xsignal_int_, dim=dim))
            signal_int = xr.merge(xsignal_int)  # .to_dataarray()
            signal_int.attrs = attrs

        return signal_int

    def convolve(self,
                 signal,
                 name='signal',
                 info={}):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param signal: xarray spectral signal to convolve, coord=wl
        :param fwhm: xarray with data=fwhm containing full width at half maximum in nm, and coords=wl
        :param info: optional parameter to feed the attributes of the output xarray
        :return:
        '''

        wl_ref = signal.wl.values
        fwhm = self.fwhm.values
        wl = self.fwhm.wl.values
        xdims = signal.dims

        if len(xdims) == 1:
            signal_int = self.convolve_(wl_ref, signal.values, wl, fwhm)
            signal_int = xr.DataArray(signal_int, name=name,
                                      coords={'wl': self.fwhm.wl.values},
                                      attrs=info)

        else:
            # to handle multidimensional xarray
            xdims = np.array(xdims)
            xdims = xdims[xdims != 'wl']

            xsignal_int = []
            for dim in xdims:
                xsignal_int_ = []
                for value, signal_ in signal.groupby(dim):
                    # print(dim, value)
                    signal_ = signal_.squeeze()
                    _ = self.convolve_(signal_.wl.values, signal_.values, wl, fwhm)
                    _ = xr.Dataset({name: (['wl'], _)},
                                   coords={'wl': wl,
                                           dim: value})
                    xsignal_int_.append(_)
                xsignal_int.append(xr.concat(xsignal_int_, dim=dim))
            signal_int = xr.merge(xsignal_int).to_dataarray()
            signal_int.attrs = info

        return signal_int
