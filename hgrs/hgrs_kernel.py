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
(:py:mod:`hgrs.spectral_sensitivity`):

.. math::

   X_i = \frac{\int X(\lambda)\, S_i(\lambda)\, d\lambda}{\int S_i(\lambda)\, d\lambda}

with a Gaussian :math:`S_i` (:py:class:`hgrs.spectral_sensitivity.Gaussian`, solar irradiance) or a
super-Gaussian of exponent 3 (:py:class:`hgrs.spectral_sensitivity.SuperGaussian`, gaseous
transmittance), whose width is derived from the full width at half maximum :math:`\Gamma_i` of the band.
The former implementation is kept in :py:mod:`hgrs.legacy.spectral`.

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

References
==========

The radiative transfer look-up tables were computed with the OSOAA code [Chami2015]_ for the OPAC aerosol
models [Hess1998]_, the absorption of the gases with the REPTRAN parameterization [Gasteiger2014]_ and the
Rayleigh optical thickness follows [Bodhaine1999]_. The SWIR sunglint correction follows the GRS
algorithm [Harmel2018]_. The optimizations use the trust-region reflective method of
:py:func:`scipy.optimize.least_squares` [Branch1999]_ and SLSQP [Kraft1988]_.

.. [Harmel2018] Harmel, T., Chami, M., Tormos, T., Reynaud, N., Danis, P.-A. (2018). Sunglint correction of the
   Multi-Spectral Instrument (MSI)-SENTINEL-2 imagery over inland and sea waters from SWIR bands.
   *Remote Sensing of Environment*, 204, 308-321. https://doi.org/10.1016/j.rse.2017.10.022
.. [Chami2015] Chami, M., Lafrance, B., Fougnie, B., Chowdhary, J., Harmel, T., Waquet, F. (2015). OSOAA: a
   vector radiative transfer model of coupled atmosphere-ocean system for a rough sea surface
   application to the estimates of the directional variations of the water leaving reflectance to
   better process multi-angular satellite sensors data over the ocean. *Optics Express*, 23(21),
   27829-27852. https://doi.org/10.1364/OE.23.027829
.. [Hess1998] Hess, M., Koepke, P., Schult, I. (1998). Optical properties of aerosols and clouds: the software
   package OPAC. *Bulletin of the American Meteorological Society*, 79(5), 831-844.
   https://doi.org/10.1175/1520-0477(1998)079<0831:OPOAAC>2.0.CO;2
.. [Gasteiger2014] Gasteiger, J., Emde, C., Mayer, B., Buras, R., Buehler, S. A., Lemke, O. (2014). Representative
   wavelengths absorption parameterization applied to satellite channels and spectral bands.
   *Journal of Quantitative Spectroscopy and Radiative Transfer*, 148, 99-115.
   https://doi.org/10.1016/j.jqsrt.2014.06.024
.. [Bodhaine1999] Bodhaine, B. A., Wood, N. B., Dutton, E. G., Slusser, J. R. (1999). On Rayleigh optical depth
   calculations. *Journal of Atmospheric and Oceanic Technology*, 16(11), 1854-1861.
   https://doi.org/10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2
.. [Branch1999] Branch, M. A., Coleman, T. F., Li, Y. (1999). A subspace, interior, and conjugate gradient method
   for large-scale bound-constrained minimization problems. *SIAM Journal on Scientific Computing*,
   21(1), 1-23. https://doi.org/10.1137/S1064827595289108
.. [Kraft1988] Kraft, D. (1988). *A software package for sequential quadratic programming*. Technical Report
   DFVLR-FB 88-28, DLR German Aerospace Center, Institute for Flight Mechanics, Köln.
'''

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
logger = logging.getLogger(__name__)


class Product():
    r'''
    Hyperspectral L1C image prepared for hGRS: spectral and processing parameters, auxiliary data and
    water masks.

    Parameters
    ----------
    l1c_obj : xarray.Dataset
        L1C image with the TOA reflectance ``Rtoa`` (wl, y, x), the band widths ``fwhm`` and the angles
        ``sza``, ``vza``, ``raa`` (see :py:mod:`hgrs.drivers`).
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

    Examples
    --------
    .. code-block:: python

       import hgrs

       l1c = hgrs.EnmapDriver().read('ENMAP01-____L1C-DT0000088121_..._V010502_...')
       prod = hgrs.Product(l1c, xcoarsen=20, ycoarsen=20)
       prod.apply_water_masks()          # Rtoa set to NaN outside water pixels
       prod.air_mass_mean, prod.wl_sunglint
    '''

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

        logger.info('Load pre-computed radiative transfer LUT')
        # pre-computed auxiliary data

        self.abs_gas_file = self.lut_tables.abs_gas_file
        self.lut_file = self.lut_tables.lut_file
        self.trans_lut_file = self.lut_tables.trans_lut_file

        self.Twv_lut = self.lut_tables.interp_twv(
            self.wl, self.sensor_description
        )

        self.load_auxiliary_data()


        logger.info(
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
        '''
        Water mask ``water_mask`` from the spectral tests (NDWI, sunglint band level and green-SWIR index,
        thresholds of the configuration), then set ``Rtoa`` to NaN outside the water pixels.
        '''
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
        Apply OmniCloudMask (Wright et al., 2025) for clouds and cloud shadows masking
        (https://github.com/DPIRD-DMA/OmniCloudMask).

        Classes of the output mask:

        - 0 = clear
        - 1 = thick cloud
        - 2 = thin cloud
        - 3 = cloud shadow

        :param rgnir: raster xarray object with the red, green and nir bands (in this order)
        :return omnimask: raster of the retrieved mask

        Examples
        --------
        As in :py:meth:`hgrs.hgrs_process.Process.execute`:

        .. code-block:: python

           rgnir = prod.raster.Rtoa.sel(wl=[670, 550, 940], method='nearest').fillna(0)
           omnimask = prod.get_omnicloudmask(rgnir)
           prod.raster['Rtoa'] = prod.raster['Rtoa'].where(omnimask == 0)

        References
        ----------
        * Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., George, R. J. (2025). Training
          sensor-agnostic deep learning models for remote sensing: achieving state-of-the-art cloud and
          cloud shadow identification with OmniCloudMask. *Remote Sensing of Environment*, 322, 114694.
          https://doi.org/10.1016/j.rse.2025.114694
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
        '''Remove the bands within the wavelength intervals ``wl_to_remove`` (list of (min, max) in nm).'''
        xarr_ = xarr.isel(x=1, y=1)
        for wls in wl_to_remove:
            wl_min, wl_max = wls
            xarr_ = xarr_.where((xarr_.wl < wl_min) | (xarr_.wl > wl_max), drop=drop)
        wl_final = xarr_.wl.values
        return xarr.sel(wl=wl_final)

    @staticmethod
    def remove_wl_dataset(xds, wl_to_remove, variable='Rtoa', drop=True):
        '''
        Remove the bands within the wavelength intervals ``wl_to_remove`` (list of (min, max) in nm),
        e.g. the strong absorption bands ``prod.wl_to_remove``.

        Examples
        --------
        .. code-block:: python

           raster = prod.remove_wl_dataset(prod.raster, prod.wl_to_remove)
        '''
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
    '''
    :py:class:`Product` with the gaseous corrections and the coarse (mega-pixel) rasters used for the
    retrievals.

    Examples
    --------
    .. code-block:: python

       prod = hgrs.Algo(l1c, xcoarsen=20, ycoarsen=20)
       prod.pressure, prod.to3c, prod.tno2c, prod.tch4c = 973.8, 6.3e-3, 3.0e-6, 9.9e-3   # from CAMS
       prod.apply_water_masks()
       prod.get_coarse_masked_raster()
       prod.get_gaseous_transmittance()     # Tg_other
       prod.other_gas_correction()          # coarse_masked_raster.Rtoa / Tg_other
    '''

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

        where :math:`\tau^*` are the normalized optical thicknesses of the gas LUT (``abs_gas_file``),
        :math:`c` the columns ``to3c``, ``tch4c``, ``tno2c``, :math:`P` the pressure (hPa) and
        :math:`\kappa` = ``coef_abs_scat``. Result in ``abs_gas_opt_thick``.

        The normalized optical thicknesses were computed with the REPTRAN parameterization
        (Gasteiger et al., 2014) at fine spectral resolution.
        '''
        gas_lut = self.lut_tables.gas_lut
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
        water vapor (``tcwv_smooth`` if present, else ``tcwv``, rounded to 0.1 kg m-2) and the water vapor
        transmittance LUT. Result in ``Twv_raster``.

        :param tcwv_raster: xarray.Dataset with ``tcwv`` (see :py:class:`WaterVapor`)
        '''
        tcwv_source = tcwv_raster.get("tcwv_smooth", tcwv_raster.tcwv)
        tcwv_vals = tcwv_source.round(1)
        tcwvs = np.unique(tcwv_vals)
        tcwvs = tcwvs[~np.isnan(tcwvs)]
        # TODO improve for air_mass raster
        Twvs = self.Twv_lut.Twv.interp(air_mass=self.air_mass_mean).interp(tcwv=tcwvs, method='linear').drop_vars('air_mass')
        self.Twv_raster = Twvs.interp(tcwv=tcwv_vals, method='nearest')
        if not behavior.PREVIOUS_BEHAVIOR:
            self.Twv_raster = self.Twv_raster.where(tcwv_vals.notnull())
        # The lookup target is an auxiliary coordinate, not a second TCWV
        # product variable. Keep it from propagating onto Rtoa/Rrs, where it
        # conflicts with the retrieved coarse-grid ``tcwv`` output variable.
        if 'tcwv' in self.Twv_raster.coords:
            self.Twv_raster = self.Twv_raster.drop_vars('tcwv')

    def get_full_resolution(self, xarr):
        '''Interpolate ``xarr`` (coarse raster) onto the full resolution grid.'''
        return xarr.interp(x=self.raster.x, y=self.raster.y)


class Solver():
    '''Common tools of the pixel-wise retrievals (uncertainties, smoothing).'''

    def __init__(self):
        pass

    @staticmethod
    def interp_linear(grid, table, value):
        r'''
        Row of ``table`` at ``value`` by linear interpolation in ``grid`` (same result as
        ``xarray.DataArray.interp``, NaN outside the grid); numpy version of the LUT interpolation for the
        cost functions, called many times per mega-pixel.

        :param grid: increasing values of the LUT coordinate
        :param table: 2D array (grid, wl)
        :param value: scalar coordinate value
        '''
        if not grid[0] <= value <= grid[-1]:
            return np.full(table.shape[1], np.nan)
        i = min(max(np.searchsorted(grid, value) - 1, 0), len(grid) - 2)
        w = (value - grid[i]) / (grid[i + 1] - grid[i])
        return (1 - w) * table[i] + w * table[i + 1]

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

        Examples
        --------
        >>> import numpy as np
        >>> from hgrs.hgrs_kernel import Solver
        >>> Solver().errFit(np.array([[2., 0.], [0., 0.5]]), 0.01)
        array([0.14142136, 0.07071068])
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

    @staticmethod
    def conv_mapping(x):
        """
        NaN-mean convolution: function for ``scipy.ndimage.generic_filter``.

        Mean of the non-NaN values of the window; if the central value is NaN, it is replaced by the mean
        of its neighbors (gap filling); it is kept if all the neighbors are NaN.

        :param x: flattened values of the footprint (the central one is ``x[len(x) // 2]``)
        :return: filtered value
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
    def _nanmean_box(image, size):
        '''
        NaN-aware mean over a rectangular window, edges extended by the nearest value.

        Same result as ``generic_filter(image, conv_mapping, footprint=np.ones(size), mode='nearest')``
        (to floating point rounding: the central value is NaN or not, the mean of the valid values of the
        window is returned, NaN if there is none), but computed with two box filters instead of a Python
        call per pixel and per window element.

        :param image: 2D array
        :param size: window shape (rows, columns)
        :return: filtered array, float64
        '''
        valid = ~np.isnan(image)
        total = ndimage.uniform_filter(np.where(valid, image, 0.), size=size, mode='nearest')
        count = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode='nearest')
        out = np.full(image.shape, np.nan)
        ok = count > 0.5 / (size[0] * size[1])  # a true count is at least 1 / (window size), the rest is rounding
        out[ok] = total[ok] / count[ok]
        return out

    @staticmethod
    @jit(nopython=True)
    def filter2d(image, weight, windows):
        r'''
        Weighted moving average of an image (numba).

        Each pixel with a positive weight is replaced by the weighted mean of its neighbours in the
        ``windows`` window :math:`W_{ij}` centered on it (pixels with a null / NaN weight are ignored):

        .. math::

           \tilde x_{ij} = \frac{\sum_{(k,l) \in W_{ij},\, w_{kl} > 0} w_{kl}\, x_{kl}}
           {\sum_{(k,l) \in W_{ij},\, w_{kl} > 0} w_{kl}}

        The window is truncated at the edges of the image; pixels with a null / NaN weight keep their
        value. All the means are computed from the input values (`image` is not modified).

        :param image: 2D array to smooth
        :param weight: 2D array of weights (e.g., number of water pixels or inverse variance), same shape
        :param windows: size of the window (rows, columns)
        :return: smoothed copy of `image`
        '''
        M, N = np.shape(image)
        Mf, Nf = windows
        Mf2 = Mf // 2
        Nf2 = Nf // 2
        threshold = 0
        result = image.copy()
        for i in range(M):
            for j in range(N):
                num = 0.0
                norm = 0.0
                if weight[i, j] > threshold:
                    for ii in range(Mf):
                        ix = i - Mf2 + ii
                        if 0 <= ix < M:
                            for jj in range(Nf):

                                iy = j - Nf2 + jj
                                if 0 <= iy < N:
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

    Examples
    --------
    .. code-block:: python

       wv = hgrs.WaterVapor(prod)
       wv.solve()
       wv.water_vapor.tcwv          # kg m-2, on the coarse grid
       prod.get_wv_transmittance_raster(wv.water_vapor)
       prod.water_vapor_correction()

    References
    ----------
    * Branch, M. A., Coleman, T. F., Li, Y. (1999). A subspace, interior, and conjugate gradient method
      for large-scale bound-constrained minimization problems. *SIAM Journal on Scientific Computing*,
      21(1), 1-23. https://doi.org/10.1137/S1064827595289108
    '''

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
        # numpy copy of the LUT for the fit: with xarray.interp at each evaluation of the residuals,
        # a fit is ~20x slower
        self.tcwv_grid = self.Twv_.tcwv.values
        self.Twv_table = self.Twv_.transpose('tcwv', 'wl').values

    def interp_tcwv(self, Twv_table, tcwv):
        r'''
        :math:`T_{wv}(\lambda; W)` by linear interpolation in ``tcwv_grid`` (same result as
        ``Twv_.interp(tcwv=tcwv)``, NaN outside the grid).

        :param Twv_table: transmittance (tcwv, wl) on ``tcwv_grid``
        :param tcwv: total column water vapor (kg m-2)
        '''
        return self.interp_linear(self.tcwv_grid, Twv_table, tcwv)

    def toa_simu(self, wl, Twv, tcwv, a, b):
        r'''Simulated reflectance :math:`T_{wv}(\lambda; W)\,(a\lambda + b)`, ``wl`` in micron

        ``Twv`` is the LUT as a DataArray (``Twv_``) or as an array (tcwv, wl) on ``tcwv_grid``.
        '''
        if isinstance(Twv, xr.DataArray):
            return Twv.interp(tcwv=tcwv, method='linear').values * (a * wl + b)
        return self.interp_tcwv(Twv, tcwv) * (a * wl + b)

    def toa_simu2(self, wl, Twv, tcwv, c0, c1, c2, c3):

        return c0 * np.exp(-c1 * wl ** -c2) * self.Twv_.interp(tcwv=tcwv).values \
            + c3 * self.wl_ ** -3 * self.Twv_.interp(tcwv=0.3 * tcwv).values

    def func(self, x, Twv, wl, y):
        return self.toa_simu(wl, Twv, *x) - y

    def func2(self, x, Twv, wl, y):
        return self.toa_simu2(wl, Twv, *x) - y

    def solve(self, x0=None):
        r'''
        Run the fit on every mega-pixel with enough water pixels (parallel processing of blocks of
        ``block_size`` mega-pixels). Result in ``water_vapor`` (``tcwv``, ``tcwv_std``).

        :param x0: first guess of :math:`(W, a, b)`; default: ``first_guess`` of the configuration
        '''
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
            water_pixel_number = self.raster.water_pixel_number.transpose('y', 'x').values
        else:
            water_pixel_number = None
        # numpy arrays: xarray indexing of each mega-pixel costs more than its fit
        spectra = np.asarray(data.transpose('wl', 'y', 'x').values)
        Twv_table = self.Twv_table
        wl_mic = self.wl_mic
        n_param = len(x0)

        global chunk_process

        def chunk_process(args):

            window_x, window_y = args

            tmp = np.ctypeslib.as_array(shared_array)
            # x0 = [20, -0.04, 0.1]
            for ix in range(window_x, min(width, window_x + block_size)):
                for iy in range(window_y, min(height, window_y + block_size)):
                    if water_pixel_number is not None:
                        if water_pixel_number[iy, ix] < pixel_threshold:
                            continue

                    y = spectra[:, iy, ix]
                    valid = ~np.isnan(y)
                    # the residual variance needs more bands than parameters
                    if valid.sum() <= n_param:
                        continue
                    y = y[valid]
                    # sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')
                    # TODO put solver parameter in self instance
                    res_lsq = least_squares(self.func, self.x0, args=(Twv_table[:, valid], wl_mic[valid], y),
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
    r'''
    Retrieval of the aerosol optical thickness :math:`\tau_{ref}` (550 nm) and of the sunglint
    amplitude :math:`B` on the coarse raster, under the constraint of non-negative water reflectance.

    :param prod: :py:class:`Algo` object corrected for gaseous absorption (other gases and water vapor)
    :param aerosol_model: OPAC aerosol model of the LUT (e.g., ``'COAV_rh70'``)
    :param first_guess: first guess of :math:`(\tau_{ref}, B)`
    :param aot550_limits: bounds of :math:`\tau_{ref}` for the optimization
    :param raster_name: name of the raster attribute of ``prod`` to process
    :param variable: reflectance variable

    Examples
    --------
    .. code-block:: python

       aero = hgrs.Aerosol(prod, aerosol_model='MACL_rh70',
                           first_guess=[0.1, 0.], aot550_limits=[0.002, 0.5])
       aero.solve()                       # aero.aero_img: aot_ref, brdfg on the coarse grid
       aero.get_atmo_parameters(prod)     # smoothing + aero.atmo_img: aot, Rtoa_diff, Tdir
       aero.atmo_img.Tdir.mean(['x', 'y']).plot()

    References
    ----------
    * Kraft, D. (1988). *A software package for sequential quadratic programming*. Technical Report
      DFVLR-FB 88-28, DLR German Aerospace Center, Institute for Flight Mechanics, Köln.
    * Hess, M., Koepke, P., Schult, I. (1998). Optical properties of aerosols and clouds: the software
      package OPAC. *Bulletin of the American Meteorological Society*, 79(5), 831-844.
      https://doi.org/10.1175/1520-0477(1998)079<0831:OPOAAC>2.0.CO;2
    '''

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

        # numpy copies of the LUTs for the fit (simulate): with xarray at each evaluation of the cost
        # function and of the constraint, the fit is much slower
        self.aot_grid = self.aot_lut.aot_ref.values
        self.aot_table = self.aot_lut.transpose('aot_ref', 'wl').values
        self.Rdiff_grid = self.Rtoa_lut.aot_ref.values
        self.Rdiff_table = self.Rtoa_lut.transpose('aot_ref', 'wl').values
        self.rot_values = self.rot.values
        self.sunglint_eps_values = self.sunglint_eps.values
        # positions of the wavelengths selected by wl_sunglint (slice or list), as in toa_simu
        self.sunglint_idx = xr.DataArray(np.arange(len(wl)), coords={'wl': self.sunglint_eps.wl}).sel(
            wl=self.wl_sunglint).values

    def simulate(self, aot_ref, BRDFg):
        r'''
        :py:meth:`toa_simu` on the numpy LUTs, at all the wavelengths ``wl``

        :param aot_ref: :math:`\tau_{ref}`
        :param BRDFg: sunglint amplitude :math:`B`
        '''
        aot = self.interp_linear(self.aot_grid, self.aot_table, aot_ref)
        Rdiff = self.interp_linear(self.Rdiff_grid, self.Rdiff_table, aot_ref)
        sunglint_corr = self.transmittance_dir(aot, self.air_mass, rot=self.rot_values) * self.sunglint_eps_values
        return Rdiff + sunglint_corr * BRDFg / sunglint_corr[self.sunglint_idx].mean()

    def residuals(self, x, idx, y):
        r'''Residuals :math:`R(\lambda) - R_{sim}(\lambda; \tau_{ref}, B)` at the wavelengths ``wl[idx]``'''
        return y - self.simulate(*x)[idx]

    def cost_residuals(self, x, idx, y):
        r''':py:meth:`cost_func` with :py:meth:`residuals`'''
        return np.sum(self.residuals(x, idx, y) ** 2)

    def constraint_residuals(self, x, idx, y):
        r''':py:meth:`constraint` with :py:meth:`residuals`'''
        return np.min(self.residuals(x, idx, y))

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
        # numpy arrays: xarray indexing of each mega-pixel costs more than its fit
        if water_pixel_number is not None:
            n_water_pixels = water_pixel_number.transpose('y', 'x').values
        spectra = np.asarray(data.transpose('wl', 'y', 'x').values)
        wl = data.wl.values

        def nearest_bands(valid, wl_target):
            # positions in wl of the valid bands nearest to wl_target (as sel(method='nearest') after dropna)
            valid_idx = np.flatnonzero(valid)
            return valid_idx[pd.Index(wl[valid_idx]).get_indexer(wl_target, method='nearest')]

        all_valid = np.ones(len(wl), dtype=bool)
        idx_atmo_all = nearest_bands(all_valid, self.wl_atmo)
        idx_non_neg_all = nearest_bands(all_valid, self.wl_non_neg)

        global chunk_process

        def chunk_process(args):
            window_x, window_y = args
            tmp = np.ctypeslib.as_array(shared_array)
            x0 = self.x0
            for ix in range(window_x, min(width, window_x + self.block_size)):
                for iy in range(window_y, min(height, window_y + self.block_size)):
                    if water_pixel_number is not None:
                        if n_water_pixels[iy, ix] < self.pixel_threshold:
                            continue
                    # x0 = self.x0
                    yfull = spectra[:, iy, ix]
                    valid = ~np.isnan(yfull)
                    # Pixels can pass the coarse water-count threshold while
                    # still having no valid spectrum after masking: leave this
                    # pixel's preinitialized retrieval values NaN.
                    if not valid.any():
                        continue
                    if valid.all():
                        idx_atmo, idx_non_neg = idx_atmo_all, idx_non_neg_all
                    else:
                        idx_atmo = nearest_bands(valid, self.wl_atmo)
                        idx_non_neg = nearest_bands(valid, self.wl_non_neg)
                    # sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')

                    cons = ({'type': 'ineq',
                             'fun': self.constraint_residuals,
                             'args': (idx_non_neg, yfull[idx_non_neg])
                             })

                    min_res = minimize(self.cost_residuals, x0,
                                       args=(idx_atmo, yfull[idx_atmo]),
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
        '''
        Smooth and gap-fill the retrieved ``aot_ref``, with the ``smoothing_method`` of the configuration:

        - ``none``: no smoothing,
        - ``weighted_local``: weighted moving average (:py:meth:`Solver.filter2d`) followed by a NaN-mean
          filter over the footprint (:py:meth:`Solver.conv_mapping`, computed with box filters),
        - ``nanmean_local``: NaN-mean filter over the footprint only.

        Result in ``aero_img['aot_ref_smoothed']``.

        :param weights: weight image (e.g., number of water pixels per mega-pixel); mega-pixels without a
            retrieved ``aot_ref`` (NaN) get a null weight
        :param windows: window of the weighted average, in mega-pixels (rows, columns); default:
            ``smoothing_window`` of the configuration
        :param mask: shape (rows, columns) of the footprint of the NaN-mean filter; default:
            ``smoothing_footprint`` of the configuration
        '''
        method = self.aerosol_state.smoothing_method
        param = self.aero_img['aot_ref'].to_numpy().astype(float)
        if method == "none":
            res = param
        elif method == "weighted_local":
            windows = self.aerosol_state.smoothing_window if windows is None else windows
            footprint = self.aerosol_state.smoothing_footprint if mask is None else mask
            # a failed retrieval must not spread NaN into the weighted mean of its neighbours
            weights = np.where(np.isnan(param), 0., weights)
            aot_ref_smoothed = self.filter2d(param, weights, np.asarray(windows))
            res = self._nanmean_box(aot_ref_smoothed, tuple(footprint))
        elif method == "nanmean_local":
            footprint = self.aerosol_state.smoothing_footprint if mask is None else mask
            res = self._nanmean_box(param, tuple(footprint))
        else:
            raise ValueError(f"Unknown aerosol smoothing method: {method}")

        self.aero_img['aot_ref_smoothed'] = xr.DataArray(res, coords=dict(y=self.aero_img.y, x=self.aero_img.x))

    def get_aot_full_resolution(self):
        '''Linear interpolation (and extrapolation) of ``aot_ref_smoothed`` at full resolution.'''
        # TODO change fill_value (extrapolate is not safe) with median of retrieval, for instance
        self.aot_ref_full = \
            self.aero_img['aot_ref_smoothed'].interp(x=self.xfull, y=self.yfull,
                                                     method='linear', kwargs={"fill_value": "extrapolate"})

    def get_atmo_parameters(self,
                            prod,
                            windows=None):
        r'''
        Rasters of the atmospheric parameters for the smoothed :math:`\tau_{ref}` (rounded to 0.001):
        spectral AOT ``aot``, diffuse reflectance ``Rtoa_diff`` and direct transmittance ``Tdir``.
        Result in ``atmo_img``.

        :param prod: :py:class:`Algo` object
        :param windows: window of the weighted average of :py:meth:`smoothing`, in mega-pixels; default:
            ``smoothing_window`` of the configuration
        '''

        wl = prod.coarse_masked_raster.wl
        weights = prod.coarse_masked_raster['water_pixel_number'].__deepcopy__().to_numpy().astype(float)

        # get LUT for desired wavelengths
        self.prepare_lut(wl)
        self.smoothing(weights, windows=windows)
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
        self.atmo_img = xr.merge([aots, Rdiffs, Tdirs], compat='no_conflicts')
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
