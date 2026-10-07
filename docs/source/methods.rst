Methods
=======

This page describes what :py:meth:`Process.execute <hgrs.hgrs_process.Process.execute>` computes, step
by step. The equations are implemented in :py:mod:`hgrs.hgrs_kernel` (see also the
:doc:`api/hgrs_kernel` overview).

Notation
--------

.. list-table::
   :widths: 25 75

   * - :math:`\lambda`
     - central wavelength of the band (nm)
   * - :math:`\theta_s,\ \theta_v,\ \phi`
     - solar zenith, viewing zenith and relative azimuth angles
   * - :math:`M`
     - two-way geometric air mass, :math:`M = 1/\cos\theta_s + 1/\cos\theta_v`
   * - :math:`L_{TOA},\ R_{TOA}`
     - top-of-atmosphere radiance and reflectance
   * - :math:`T_g,\ T_{wv}`
     - transmittances of the absorbing gases other than water vapor, and of water vapor
   * - :math:`W`
     - total column water vapor (kg m\ :sup:`-2`)
   * - :math:`\tau_{ref}`
     - aerosol optical thickness (AOT) at 550 nm
   * - :math:`\tau_a(\lambda),\ \tau_R(\lambda)`
     - aerosol and Rayleigh optical thicknesses
   * - :math:`P,\ P_0`
     - CAMS surface pressure and reference pressure (:math:`P_0` = 1013.25 hPa)
   * - :math:`\varepsilon(\lambda)`
     - mean spectral shape of the sunglint reflectance
   * - :math:`B`
     - sunglint amplitude (sunglint reflectance in the 2150-2250 nm bands)
   * - :math:`\Lambda_{wv},\ \Lambda_g,\ \Lambda_a,\ \Lambda_+`
     - wavelength sets of the water vapor fit, of the sunglint, of the aerosol fit and of the
       non-negativity constraint (`Wavelength sets`_)

Processing steps
----------------

1. **Level-1 image** (`TOA reflectance`_). EnMAP L1C or PRISMA L1 (with the angles of the L2C product,
   and geoprojection onto a regular longitude-latitude grid) are read by
   :py:class:`~hgrs.driver.Driver`.
2. **Ancillary data.** The CAMS file, taken at the image center and the closest time, gives the surface
   pressure, the O\ :sub:`3`, NO\ :sub:`2` and CH\ :sub:`4` columns and the spectral AOD.
3. **Aerosol model.** The OPAC model :math:`m` whose spectral AOD is closest to the CAMS one is
   selected:

   .. math::

      \hat m = \underset{m}{\operatorname{argmin}} \sum_{\lambda}
      \left| \frac{\tau_{CAMS}(\lambda)}{\tau_{CAMS}(550)} - \tau_{a,m}(\lambda; \tau_{ref} = 1) \right|,
      \qquad \lambda \in \{469, 550, 670, 865, 1240\}\ \text{nm}

4. **Masks.** Clouds and cloud shadows are removed with OmniCloudMask (bands at 670, 550 and 940 nm),
   then the water pixels are selected (`Water pixels`_).
5. **Coarse raster.** The water pixels are averaged over mega-pixels of 20 x 20 pixels; a mega-pixel is
   processed when at least 20 % of its pixels are water pixels.
6. **Gaseous absorption** other than water vapor (`Gaseous transmittance`_).
7. **Water vapor** retrieval on the coarse raster and correction (`Water vapor`_). The bands of strong
   absorption (935-967, 1105-1170, 1320-1490, 1778-2033 and 2465-2550 nm) and those with
   :math:`T_g T_{wv} < 0.5` are removed.
8. **Aerosol optical thickness and sunglint** on the coarse raster (`Aerosol retrieval`_), then
   smoothing and gap filling of :math:`\tau_{ref}`.
9. **Full resolution correction**, by blocks of 256 x 256 pixels (`Remote-sensing reflectance`_).
10. **Output.** :py:meth:`~hgrs.hgrs_process.Process.write_output` writes the product (see
    :doc:`usage`).

TOA reflectance
---------------

.. math::

   R_{TOA}(\lambda) = \frac{\pi\, L_{TOA}(\lambda)}{F_0(\lambda)\, \cos\theta_s}

where :math:`F_0` is the TSIS-1 solar irradiance (Coddington et al., 2021), corrected for the
Earth-Sun distance on the day of the year (DOY)

.. math::

   F_0 \leftarrow F_0 \left(\frac{d_0}{d}\right)^2, \qquad
   \left(\frac{d_0}{d}\right)^2 = 1.00011 + 0.034221 \cos\Theta + 0.00128 \sin\Theta
   + 0.000719 \cos 2\Theta + 0.000077 \sin 2\Theta, \quad \Theta = \frac{2\pi\, \mathrm{DOY}}{365}

and convolved with the spectral response of each band :math:`i`, modeled as a Gaussian of full width
at half maximum :math:`\Gamma_i`:

.. math::

   X_i = \frac{\int X(\lambda)\, S_i(\lambda)\, d\lambda}{\int S_i(\lambda)\, d\lambda},
   \qquad S_i(\lambda) = \exp\left(-\frac{(\lambda - \lambda_i)^2}{2\sigma_i^2}\right)

Water pixels
------------

With the TOA reflectances averaged over 540-570 nm (:math:`R_G`), 850-882 nm (:math:`R_{NIR}`),
1580-1650 nm (:math:`R_{1600}`) and 2150-2250 nm (:math:`R_{2200}`), the water pixels satisfy

.. math::

   \frac{R_G - R_{NIR}}{R_G + R_{NIR}} > 0.01, \qquad
   \frac{R_G - R_{1600}}{R_G + R_{1600}} > 0.1, \qquad
   R_{2200} < 0.11

Gaseous transmittance
---------------------

The optical thickness of the gases other than water vapor is built from normalized absorption
spectra :math:`\tau^*` scaled by the CAMS columns :math:`c` and the pressure:

.. math::

   \tau_g(\lambda) = c_{O_3}\tau^*_{O_3} + c_{CH_4}\tau^*_{CH_4} + c_{NO_2}\tau^*_{NO_2}
   + \frac{P}{1000}\left[\tau^*_{CO} + \kappa\left(\tau^*_{CO_2} + \tau^*_{O_2} + \tau^*_{O_4}\right)\right]

with :math:`\kappa` = ``coef_abs_scat`` = 0.35. The transmittance, for the mean air mass
:math:`\bar M` of the image, is convolved with the band responses and the TOA reflectance is corrected:

.. math::

   T_g(\lambda) = \exp\left(-\bar M\, \tau_g(\lambda)\right), \qquad
   R(\lambda) = \frac{R_{TOA}(\lambda)}{T_g(\lambda)}

Water vapor
-----------

In each mega-pixel, the reflectance over :math:`\Lambda_{wv}` (800-1300 nm) is fitted by a linear
background attenuated by water vapor:

.. math::

   R(\lambda) \simeq T_{wv}(\lambda; W, \bar M)\, (a\lambda + b), \qquad \lambda\ \text{in µm}

where :math:`T_{wv}` is interpolated in a pre-computed LUT. :math:`(W, a, b)` are retrieved by
non-linear least squares within :math:`[0, 60] \times [-10, 1] \times [0, 1]`; the uncertainty of
:math:`W` is

.. math::

   \sigma_W = \sqrt{\left[(J^T J)^{-1}\right]_{WW}\, s^2},
   \qquad s^2 = \frac{\sum_j r_j^2}{n - 3}

with :math:`J` the Jacobian of the residuals :math:`r` and :math:`n` the number of bands. The
reflectance is then divided by :math:`T_{wv}(\lambda; W)`, with :math:`W` of the nearest mega-pixel.

.. _methods-aerosol:

Aerosol retrieval
-----------------

The atmospheric diffuse reflectance (Rayleigh + aerosol) is interpolated in the radiative transfer
LUT of the selected model, for the mean geometry of the image:

.. math::

   R_{diff}(\lambda; \tau_{ref}) = \frac{I(\lambda; \tau_{ref}, \theta_s, \theta_v, \phi_{LUT})}{\cos\theta_s},
   \qquad \phi_{LUT} = (180^\circ - \phi) \bmod 360^\circ

and the direct transmittance is

.. math::

   T_{dir}(\lambda; \tau_{ref}) = \exp\left[-\left(\tau_R(\lambda)\frac{P}{P_0} + \tau_a(\lambda; \tau_{ref})\right) M\right]

The sunglint has the spectral shape :math:`\varepsilon(\lambda)` of the Fresnel reflectance and is
attenuated by the direct transmittance, so that the TOA signal of a water pixel is modeled as

.. math::

   R_{sim}(\lambda; \tau_{ref}, B) = R_{diff}(\lambda; \tau_{ref})
   + B\, \frac{T_{dir}(\lambda)\, \varepsilon(\lambda)}{\left\langle T_{dir}\, \varepsilon \right\rangle_{\Lambda_g}}

where :math:`\langle\cdot\rangle_{\Lambda_g}` is the average over the 2150-2250 nm bands. In each
mega-pixel, :math:`(\tau_{ref}, B)` are retrieved by constrained optimization (SLSQP):

.. math::

   (\hat\tau_{ref}, \hat B) = \underset{\tau_{ref},\, B}{\operatorname{argmin}}
   \sum_{\lambda \in \Lambda_a} \left[R(\lambda) - R_{sim}(\lambda; \tau_{ref}, B)\right]^2
   \quad \text{subject to} \quad
   \min_{\lambda \in \Lambda_+} \left[R(\lambda) - R_{sim}(\lambda; \tau_{ref}, B)\right] \geq 0

The fit uses the NIR-SWIR bands of :math:`\Lambda_a`, where the water is black, while the constraint
imposes that the water reflectance stays non-negative in the visible-NIR bands of :math:`\Lambda_+`.
The bounds are :math:`B \in [0, 1.3]` and

.. math::

   0.002 \leq \tau_{ref} \leq \bar\tau_{CAMS}(550) + 2 \max\left(s_{CAMS},\ 0.2\, \bar\tau_{CAMS}(550) + 0.05\right)

with :math:`s_{CAMS}` the standard deviation of the CAMS AOD; the first guess is
:math:`(\bar\tau_{CAMS}(550), 0)`.

The retrieved :math:`\tau_{ref}` is smoothed with a weighted moving average (weights: number of water
pixels of the mega-pixels)

.. math::

   \tilde\tau_{kl} = \frac{\sum_{(i,j) \in W_{kl}} N_{ij}\, \tau_{ij}}{\sum_{(i,j) \in W_{kl}} N_{ij}}

followed by a 3 x 3 NaN-mean filter that fills the gaps.

Remote-sensing reflectance
--------------------------

The smoothed :math:`\tau_{ref}` gives the rasters :math:`R_{diff}(\lambda)` and :math:`T_{dir}(\lambda)`,
linearly interpolated at full resolution. In each pixel:

.. math::

   R_{corr}(\lambda) = \frac{R_{TOA}(\lambda)}{T_g(\lambda)\, T_{wv}(\lambda)} - R_{diff}(\lambda)

the sunglint amplitude is estimated in the SWIR, where the water-leaving signal is negligible,

.. math::

   B = \left\langle \frac{R_{corr}(\lambda)}{T_{dir}(\lambda)\, \varepsilon(\lambda)} \right\rangle_{\Lambda_g}

and the remote-sensing reflectance is

.. math::

   R_{rs}(\lambda) = \frac{R_{corr}(\lambda) - T_{dir}(\lambda)\, \varepsilon(\lambda)\, B}
   {\pi\, T_{\downarrow}(\lambda; \theta_s)\, T_{\downarrow}(\lambda; \theta_v)^{1.05}}

where :math:`T_{\downarrow}` is the total (direct + diffuse) transmittance of the downwelling irradiance
from the LUT, for the mean geometry and the mean retrieved :math:`\tau_{ref}`.

Wavelength sets
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Set
     - Attribute
     - Wavelengths (nm)
   * - :math:`\Lambda_{wv}`
     - ``wl_water_vapor``
     - 800-1300
   * - :math:`\Lambda_g`
     - ``wl_sunglint``
     - 2150-2250
   * - :math:`\Lambda_a`
     - ``wl_atmo``
     - 1000, 1050, 1075, 1100, 1200, 1300, 1600, 1650, 1700, 2150, 2200, 2250
   * - :math:`\Lambda_+`
     - ``wl_non_neg``
     - 430, 490, 560, 650, 750, 800, 865, 1020

(attributes of :py:class:`~hgrs.hgrs_kernel.Product`; the closest bands of the sensor are used).

References
----------

The radiative transfer look-up tables were computed with the OSOAA code (Chami et al., 2015) for the
OPAC aerosol models (Hess et al., 1998), the gaseous absorption with REPTRAN (Gasteiger et al., 2014),
and the Rayleigh optical thickness follows Bodhaine et al. (1999). The SWIR sunglint correction
derives from the GRS algorithm (Harmel et al., 2018).

* Harmel, T., Chami, M., Tormos, T., Reynaud, N., Danis, P.-A. (2018). Sunglint correction of the
  Multi-Spectral Instrument (MSI)-SENTINEL-2 imagery over inland and sea waters from SWIR bands.
  *Remote Sensing of Environment*, 204, 308-321. https://doi.org/10.1016/j.rse.2017.10.022
* Chami, M., Lafrance, B., Fougnie, B., Chowdhary, J., Harmel, T., Waquet, F. (2015). OSOAA: a
  vector radiative transfer model of coupled atmosphere-ocean system for a rough sea surface
  application to the estimates of the directional variations of the water leaving reflectance to
  better process multi-angular satellite sensors data over the ocean. *Optics Express*, 23(21),
  27829-27852. https://doi.org/10.1364/OE.23.027829
* Hess, M., Koepke, P., Schult, I. (1998). Optical properties of aerosols and clouds: the software
  package OPAC. *Bulletin of the American Meteorological Society*, 79(5), 831-844.
  https://doi.org/10.1175/1520-0477(1998)079<0831:OPOAAC>2.0.CO;2
* Gasteiger, J., Emde, C., Mayer, B., Buras, R., Buehler, S. A., Lemke, O. (2014). Representative
  wavelengths absorption parameterization applied to satellite channels and spectral bands.
  *Journal of Quantitative Spectroscopy and Radiative Transfer*, 148, 99-115.
  https://doi.org/10.1016/j.jqsrt.2014.06.024
* Bodhaine, B. A., Wood, N. B., Dutton, E. G., Slusser, J. R. (1999). On Rayleigh optical depth
  calculations. *Journal of Atmospheric and Oceanic Technology*, 16(11), 1854-1861.
  https://doi.org/10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2
* Coddington, O. M., Richard, E. C., Harber, D., et al. (2021). The TSIS-1 hybrid solar reference
  spectrum. *Geophysical Research Letters*, 48(12), e2020GL091709. https://doi.org/10.1029/2020GL091709
* Spencer, J. W. (1971). Fourier series representation of the position of the sun. *Search*, 2(5),
  172.
* Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., George, R. J. (2025). Training
  sensor-agnostic deep learning models for remote sensing: achieving state-of-the-art cloud and
  cloud shadow identification with OmniCloudMask. *Remote Sensing of Environment*, 322, 114694.
  https://doi.org/10.1016/j.rse.2025.114694
* Kraft, D. (1988). *A software package for sequential quadratic programming*. Technical Report
  DFVLR-FB 88-28, DLR German Aerospace Center, Institute for Flight Mechanics, Köln.
* Branch, M. A., Coleman, T. F., Li, Y. (1999). A subspace, interior, and conjugate gradient method
  for large-scale bound-constrained minimization problems. *SIAM Journal on Scientific Computing*,
  21(1), 1-23. https://doi.org/10.1137/S1064827595289108
* Cogliati, S., Sarti, F., Chiarantini, L., et al. (2021). The PRISMA imaging spectroscopy mission:
  overview and first performance analysis. *Remote Sensing of Environment*, 262, 112499.
  https://doi.org/10.1016/j.rse.2021.112499
* Storch, T., Honold, H.-P., Chabrillat, S., et al. (2023). The EnMAP imaging spectroscopy mission
  towards operations. *Remote Sensing of Environment*, 294, 113632.
  https://doi.org/10.1016/j.rse.2023.113632
