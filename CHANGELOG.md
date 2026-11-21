# Changelog

All notable changes to hGRS are listed here, most recent first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). Dates are those of the commits that set the version.

## [Unreleased]

## [2.1.0] - 2026-11-21

### Added
- Solar spectrum of the L1 drivers (defaults unchanged, products identical; branch `feature/solar_spectrum`):
  setting `product: solar_spectrum` (`tsis` by default, `thuillier2003`, `thuillier`, `gueymard`, `kurucz`), or
  the attribute `solar_spectrum` of a driver, for the conversion of the L1 radiance into reflectance by all the
  drivers (EnMAP, PRISMA, EMIT, Tanager, Hyperion; `L1Driver.solar_irradiance`). New spectrum `thuillier2003`
  (Thuillier et al. 2003 composite, `data/aux/Thuillier_2003_0.3nm.dat`, shipped but unused so far): the
  spectrum of the Sentinel-2 L1C solar irradiances (within 0.02 % over the MSI responses) and of OLCI.
  `SolarIrradiance.get(name)` and `SOLAR_SPECTRA` (name -> reference). `F0` of the L1 product now carries
  `solar_spectrum`, `solar_spectrum_reference`, `mission_solar_spectrum` and `radiance_calibration`; each
  driver declares how its mission calibrates the radiance. EnMAP is calibrated with its sun diffuser and the
  Fontenla et al. (2011) spectrum (de los Reyes et al. 2022), not available in hGRS: its driver logs a warning
  when converting with another spectrum. On an EnMAP scene, TSIS-1 / Thuillier 2003 over the bands: +0.6 % at
  443 nm, +3.2 % at 560 nm, -0.8 % at 865 nm, -3.2 / -5.3 % at 1610 / 2190 nm.

- Water vapor, coupled model (`gas_coupling: path_length`), experimental constraints (branch
  `feature/abs_scat_coupling`; defaults unchanged, products identical): `water_vapor: surface: non_negative`
  replaces the surface term a wl + b (any sign) by the line between its values s0, s1 >= 0 at the ends of the
  fitting range, fitted to convergence (`gtol` 1e-15: with residuals ~1e-5 the default stops the fit early);
  `path_scale_limits` bounds the scale factor k of the LUT path reflectance ([0, 5] in 2.0.0); `offset:
  saturated_bands` subtracts the median reflectance of the bands of `offset_wavelengths_nm` (1360-1395 nm) in
  each mega-pixel. With a surface term of any sign the fit can put the path term above the data (k R_p up to
  2x the signal) and a negative surface term under it, whose deeper bands make the sum shallower: W is then too
  high (RdP-EsNM 33-47 against 6-14 kg m-2 from AERONET, Lake Okeechobee 52 against 9, Banana River 29
  against 6). `path_slope: free` (with `surface: non_negative`) multiplies the path reflectance by wl^-dn, dn in
  `path_slope_limits` ([-4, 4]): with the LUT shape of the first-guess aerosol model and AOT, the surface term
  takes 15-40 % of the signal at 1040 nm over dark water to correct the slope of the continuum, and its
  direct-beam bands lower W (0.65 of AERONET, median of 9 dark matchups; 0.79 with the free slope).
  `path_scale_limits` must leave room for a first-guess AOT far from the truth: with [0.5, 2] W reaches 60 at
  Banana River (CAMS AOT550 0.21-0.37 against 0.05-0.08 from AERONET, LUT path reflectance ~3x the signal, k
  needs ~0.3 and the fit lowers the model by absorption), and [0.2, 3] gives 9.3-9.5 against 14.7-15.9 (3
  other matchups change by 0.5-7 kg m-2, MVCO and Chesapeake Bay). AERONET-OC validation (64 matchups,
  `surface: non_negative`, `path_slope: free`, `path_scale_limits: [0.2, 3]`): TCWV bias -1.8 kg m-2, RMSD 5.4
  (2.0.0: -0.8, 10.3; CAMS prior: +0.8, 3.8), 0.73 of AERONET below 10 kg m-2.
- Pressure scaling of the Rayleigh scattering of the path reflectance (current behavior; not with
  `previous_behavior`; branch `feature/rayleigh_pressure`): the LUTs are OSOAA runs with the Rayleigh optical
  thickness of 1015.2 hPa (`LUT_PRESSURE`), and the path reflectance was not scaled with the surface pressure
  (+/-2.7e-3 at 400 nm, +/-1.2e-3 at 500 nm for +/-20 hPa, +0.3e-3 at 400 nm at 1013 hPa). `Aerosol.prepare_lut` adds
  `dRray` = R_ray [(P/P_LUT)^D - 1] (`rayleigh_pressure_delta`, D = dln R_ray / dln tau_r from the wavelength nodes of
  the LUT, 0.90 at 400 nm) to `Rtoa_lut` (aerosol fit and atmospheric image), and the final transmittance of the Rrs
  is multiplied by exp(-dtau_r (1/2mu_s + 1/2mu_v)); validated with OSOAA runs at +/-20 hPa in radcalnet_oc
  (`scripts/rayleigh_pressure`).
- EnMAP, experimental: correction of the radiometric mismatch of the VNIR and SWIR detectors over dark water,
  `product: enmap_detector_offset: none | vnir_ramp | swir_ramp` (`none` by default, products unchanged). The
  offset SWIR - VNIR is the median over the water pixels of the scene at the VNIR band nearest to 990 nm
  (detector overlap); `vnir_ramp` adds it to the VNIR, ramped from 0 at 700 nm to 1 at 990 nm, `swir_ramp`
  subtracts it from the SWIR, tapered to 0 at 2200 nm (`hgrs.drivers.enmap.detector_offset`; attribute
  `detector_offset_990`). Over dark water the SWIR detector reads 4e-3 to 1.3e-2 more than the VNIR one.
- Aerosol retrieval, experimental: `aerosol: prior: none | cams` (Tikhonov term towards the CAMS AOD(550), with
  the sigma of the bounds of the fit), `nonnegativity: hard | soft` (non-negativity of the water reflectance as
  the constraint of the fit, or as a one-sided quadratic penalty of scale `nonnegativity_sigma`) and
  `reflectance_sigma` (weight of the data term); `AerosolModel.cost_regularized`. The defaults (`none`, `hard`)
  give the 2.0.0 fit. Over dark water, where the non-negativity of the EnMAP VNIR bands cannot be met and the
  AOT ends at its lower bound, a soft non-negativity lets the SWIR bands and CAMS set the AOT (Lucinda scene:
  98 % of the mega-pixels at the bound with the 2.0.0 fit, 4 % with `soft`, `nonnegativity_sigma: 2e-3` and the
  CAMS prior).
- Water vapor retrieval, experimental: `water_vapor: prior: none | cams` (Tikhonov term towards the CAMS column,
  `sigma_W = max(prior_sigma_relative * W_CAMS, prior_sigma_min)`, defaults 0.1 and 2 kg m-2) and
  `reflectance_sigma` (weight of the data term), in both the coupled and decoupled fits
  (`WaterVapor.regularization`). `prior: none` (default) gives the 2.0.0 fit. On 22 AERONET-OC matchups chosen
  among the largest water vapor errors, the MAPD of the column drops from 36 % to 4 %, as for CAMS alone.
- Database of the EnMAP and PRISMA images, `hgrs.catalog` and the command `hgrs_catalog`: one copy of each
  product by collection (`l1c`, `l1`, `l2c`, `l2a-hgrs/v<version>`, `l2w-acolite`, ...) and acquisition date,
  producer and version read from the products; migration of existing folders as a reviewable plan (`plan`,
  `apply`, `undo`: renames only, redundant copies to `_trash/`, former paths kept as symlinks in `legacy/`);
  static STAC catalogue (layout of grstbx / GRSdriver.stac), GeoParquet index, views by site (former folders
  and point sites inside the footprints), `search` and `ingest` of new downloads. Optional dependencies:
  `pip install hGRS[catalog]` (geopandas, pyarrow, shapely) for the index and the search.

### Changed
- Faster spectral convolution, same products within 1e-14 (not with `previous_behavior`): the band responses are
  computed once per wavelength grid as a sparse matrix of the trapezoidal weights
  (`GenericSpecSen.convolution_matrix`), and a convolution is one matrix product instead of a loop over the bands
  (60 water vapor columns on the 21687 wavelengths of the gaseous LUT, 224 bands: 0.64 s to 21 ms). The water vapor
  transmittance table is built and convolved at the two air-mass nodes around the mean air mass of the scene only
  (`LUTTables.interp_twv(..., air_mass)`, `Product.Twv_lut` computed on first use), instead of the 54 nodes, ~300 MB
  less at peak: its linear interpolation at the air mass is unchanged (3e-15), 0.2 s instead of 1.7-12 s.
- Lower memory use, same products (bit for bit): the TOA reflectance cube is read once into a single array, and
  masked (water, clouds, bands of strong gaseous absorption) and corrected for the other gases in place; the
  coarse raster is averaged by strips of mega-pixels; GDAL block cache of 256 MB for EnMAP (`GDAL_CACHEMAX`, kept
  if set). Peak memory for a full EnMAP scene: 14.9 to 10.1 GB.
- TOA reflectance `Rtoa` processed in float32 (float64 with `previous_behavior`): peak memory for a full EnMAP
  scene 10.1 to 8.0 GB. The products change at the float32 rounding level only: `Rrs` by 1.5e-8 at most (median
  relative difference 1.2e-7), `aot_ref` by 6e-7, `tcwv` by 6e-4 kg m-2 for 99 % of the mega-pixels (0.015 at
  most); masks unchanged.

### Fixed
Each fix changes the products; `previous_behavior` restores the former processing.
- Gaussian band response: sigma = FWHM / (2 sqrt(2 ln 2)) (`gaussian_fwhm_to_sigma`, also used by
  `SolarIrradiance.Gamma2sigma`); the former sigma was sqrt(2) too large (0.60 FWHM).
- EnMAP: the solar irradiance F0 is convolved with the band response once; the former two-stage resampling
  applied it twice (up to ~2.7 % error on F0 near the Fraunhofer lines).
- PRISMA: in the VNIR / SWIR overlap (~943-956 nm) the VNIR bands from the first SWIR band on are dropped, so
  that each wavelength comes from one detector, as for EnMAP.
- Aerosol LUT: radiances interpolated in wavelength quadratically in log-log (power law of the wavelength) instead
  of linear space (leave-one-out on the OSOAA LUT: ~1 % error instead of ~10 % in the visible).
- Water vapor, coupled model (`gas_coupling: path_length`): the path reflectance of the TOA LUT at the first-guess
  AOT (`WaterVapor.path_reflectance_first_guess`) is now that of the aerosol fit, interpolated in wavelength in
  log-log space and with the Rayleigh scattering at the surface pressure; it was still interpolated in linear space
  without the pressure term. Between the LUT nodes 1000 and 1600 nm the former shape was off by -2.5 to +0.8 % over
  931-1060 nm at AOT 0.05 (-1.1 to +0.3 % at 0.2; ARCT and MACL models), up to 0.8 % once the scale k and a free
  path slope absorb it; the pressure term is negligible there. Found by the comparison with radcalnet_oc 1.2.
- OmniCloudMask: run with the NIR at 865 nm and at 940 nm, a pixel is masked only where both runs flag it. With
  940 nm alone the input depended on the water vapour and gave wrong shadows; with 865 nm alone clear dark water
  was flagged as thin cloud (AAOT 2025-07-04 masked entirely). On 89 AERONET-OC crops: 5.0 % of the water
  masked instead of 9.6 % (865) / 7.1 % (940), station pixels fully masked on 2 crops instead of 6 / 4.
- Water vapor transmittance: a clear `ValueError` when the water vapor retrieval is NaN everywhere (no water
  pixel left after the cloud and water masks) instead of numpy's "zero-size array to reduction operation fmin".
- `SolarIrradiance.thuillier` is the ATLAS3 spectrum (`ref_atlas_thuillier3.nc`, Thuillier et al. 2004), 1.4 %
  above the Thuillier et al. (2003) composite from 500 to 2400 nm: docstring and reference corrected (it
  cited the 2003 composite); documentation only, products unchanged.

## [2.0.0] - 2026-11-13

### Added
- Coupling of the gaseous absorption with the scattering, `hgrs.gas_coupling` (ported from
  [radcalnet_oc](https://github.com/Tristanovsk/radcalnet_oc), commit d4f6c6f): the path reflectance and the
  water-leaving signal have their own gaseous transmittance, from the photon path-length model
  $T = (1 + s^2 \eta \bar M \tau)^{-1/s^2}$ fitted on SOS-ABS coupled simulations, in the O2 and H2O windows of
  684-990 nm (`data/aux/gas_coupling_ppld.csv`); the sun glint keeps the direct-beam transmittance. In each window
  $\tau$ is the total absorption optical thickness of the channel (all the gases together, as in the fit: the
  coupled transmittance of a channel is not the product of those of its gases). Used in the water vapor retrieval
  (path reflectance of the TOA LUT at the first-guess AOT with a free amplitude, first guess of the column from
  CAMS), in the aerosol fit and in the final correction,
  $R_{rs} = [R_{toa}/T_d - f_{path} R_{diff} - R_{dir}]\, f_w / \pi$. Settings
  `atmospheric_correction: product: gas_coupling: path_length | decoupled` and `gas_coupling_table` (CSV of the
  coefficients; `null`, the table of the package).
- Line-by-line O2 absorption (HITRAN, table `data/lut/o2_lbl_tau.nc` of radcalnet_oc), in place of the REPTRAN
  table. Setting `atmospheric_correction: product: o2_model: lbl_o2b | lbl | reptran`: `lbl_o2b` line by line in
  the O2-B band (683-702 nm) and REPTRAN in the O2-A band (755-775 nm), `lbl` line by line in both. Against the
  SOS-ABS CKD, the band-mean O2 absorption of REPTRAN is 42-72 % too strong in the O2-B band and that of the
  line-by-line table 15-18 % too weak in the O2-A band (air masses 1-3.5).
- Research notebooks *Coupling of absorption and scattering* (transmittance of each term against SOS-ABS, water vapor
  retrieval, O2 bands) and their helpers, in `docs/source/tutorials/research`.
- Validation scripts at the AERONET-OC stations with EnMAP, `validation/aeronet_oc` (matchup inventory, extraction,
  processing with several versions, radiometric closure in the gaseous absorption bands), and
  `utils/fit_gas_coupling_full.py`, which runs the fit of radcalnet_oc over 364-2500 nm and writes the table
  `gas_coupling_ppld_full.csv` (not used by default).

### Changed
- The default settings use `gas_coupling: path_length`, `o2_model: lbl_o2b` and a water vapor fitting range
  `water_vapor: wavelengths_nm: [931, 1060]` (was 800-1300 nm): SWIR detector of EnMAP only, 940 nm band (long-wave
  half) and its continuum, because the VNIR and SWIR reflectances disagree around 900 nm by up to 50 %, which biases
  a fit across them. Over dark water the decoupled model underestimates the water vapor column (by a factor 3 to 5
  at the AERONET-OC stations), the coupled one recovers it.
- The products change in the O2 and H2O bands of 684-990 nm and the retrieved water vapor column changes; they are
  unchanged elsewhere. In the coupled windows the factor `gas_absorption_scattering_coefficient` of O2, O4 and CO2
  is 1. `previous_behavior: true` (`gas_coupling: decoupled`, `o2_model: reptran`, 800-1300 nm) gives the results
  of 1.2.0.
- On two dark-water EnMAP scenes (Lucinda 2026-05-31, Brasil 2024-06-12 tile 001), `previous_behavior` (1.2.0) ->
  2.0.0 defaults: TCWV 1.0 -> 12.2 and 2.7 -> 17.9 kg m-2 (CAMS 11.0 and 14.1); median residual band depth of Rrs
  (sr-1): 764 nm -5.0e-4 -> -5e-5 and -8.8e-4 -> -9e-5, 720 nm -4.0e-4 -> -1.2e-4 and -6.2e-4 -> -2e-5, 817 nm
  -1.1e-4 -> +1.0e-4 and -2.1e-4 -> +3.0e-4, 686 nm +8e-5 -> -1.0e-4 and +7e-5 -> -1.4e-4; the 941 nm band is
  dropped (gaseous transmittance below 0.5 with the retrieved column); median Rrs at 443, 560, 665 and 865 nm within
  7e-5. On both scenes the AOT of the retrieval stays at its lower bound, with and without the coupling: over dark
  water the TOA reflectance of the EnMAP VNIR detector beyond ~720 nm is below the Rayleigh path reflectance (by
  up to 3.5e-3 at 890 nm), which the non-negativity constraint of the aerosol fit (750, 800, 865 nm) cannot
  accommodate. In the VNIR/SWIR overlap (975-990 nm) the SWIR detector reads 3e-3 to 1e-2 more than the VNIR one
  over dark water at the same wavelength, on all the 17 scenes checked (L1C V010502 and V010506), while the two
  agree to a few percent over land.

## [1.2.0] - 2026-11-12

### Added
- User configuration file `~/.config/hgrs/config.yml` (or `$XDG_CONFIG_HOME/hgrs/config.yml`),
  read at run time on top of the packaged defaults of `hgrs/config/default_config.yml`; the `HGRS_CONFIG` and
  `HGRS_DATA_ROOT` environment variables select another file or the LUT folder.
- `hgrs_config` command to print the configuration in use and check the LUT files (exit code 1 if
  one is missing); `hgrs_config --init /path/to/lut` creates the user configuration file with the LUT
  folder and, commented, the default settings.
- The processing checks the LUT files before reading the image, with a message giving the missing
  files and how to set their folder.
- Default folder of the LUTs, used when `data_root` is not set: `~/.local/share/hgrs/lut` on Linux
  (user data folder given by `platformdirs`), so that copying the LUTs there needs no configuration;
  `hgrs_config` prints the LUT folder in use and where it is set, and `hgrs_config --init` without a
  folder creates the default one. `platformdirs` is a new dependency.
- `previous_behavior` setting of the user configuration file and `HGRS_PREVIOUS_BEHAVIOR`
  environment variable, to select the former parameters and algorithms without editing the package.
- Configuration page of the documentation: quick start, origin of the settings, annotated default
  settings, examples, former behavior and troubleshooting.
- Read the Docs documentation: physical model and equations in the docstrings, Methods,
  Installation and History pages, API reference, runnable examples (doctests) and references.
- Processing Chain Description page of the documentation (granule, scheduling, dataflow diagram,
  inputs, outputs, return codes, log format, resources, data types), following the processing chain
  documentation template.
- Tutorial notebooks on an EnMAP image of Lake Trasimeno: image processing, batch processing and
  visualization.
- Tutorial notebook on the visualization of a time series of hGRS L2A products with grstbx
  (true-colour composite, area of interest, multi-date spectra and time series, interactive viewers).
- Tutorial notebook on a PRISMA image: the L1 TOA reflectance and the ASI L2C reflectance interpolated onto
  the grid of the hGRS L2A product, true-colour composites before and after correction, retrieved parameters,
  TOA and $R_{rs}$ spectra of points of interest, spectral transect, comparison with the L2C reflectance.
- Drivers and commands for EMIT L1B (`hgrs_emit`), Hyperion L1R (`hgrs_hyperion`) and Tanager
  (`hgrs_tanager`) images, selected with the new `sensor` parameter of `Process.execute`, with the
  `geoproject` option to keep the native pixel grid (A. Lavard-Meyer).
- Configuration package `hgrs.config`: dataclasses of the settings of the masks, water vapor and
  aerosol retrievals (`atmospheric_correction` section of the configuration file), scene metadata
  and sensor description; the switch `hgrs.config.behavior.PREVIOUS_BEHAVIOR` selects the former
  parameters (`old_config.yml`) and algorithms (A. Lavard-Meyer).
- PRISMA: the land mask of the L2C product excludes land and whitecap pixels from the retrievals
  (A. Lavard-Meyer).
- `cloud_mask` and `water_mask` in the L2A product; reanalysis CAMS files (`valid_time`) are read.
- Spectral response functions in `hgrs.spectral_sensitivity`; the former implementation is kept in
  `hgrs.legacy.spectral` (A. Lavard-Meyer).
- New hGRS logo for the light and dark modes of the documentation.
- `docs/make.bat` to build the documentation on Windows, and `.gitattributes` to check out `.bat`
  files with CRLF line endings.

### Changed
- **Products differ from 1.1.2** with the default settings, because of the reworked water vapor and
  aerosol solvers (A. Lavard-Meyer, see below): the water vapor transmittance LUT is built from the
  gas absorption and convolved with the spectral response of the sensor (instead of interpolating a
  precomputed LUT), the spectral response uses a slightly different (super-)Gaussian formula, and the
  mega-pixels without a retrieved TCWV are filled. On a 400 x 400 pixel water subset of an EnMAP
  L1C image (Lucinda, 2026-05-31): TCWV -13 %, Rrs -6.8 % at 418 nm, -0.8 to -1.8 % from 459 to
  577 nm, and a bias below 3e-5 sr-1 beyond 600 nm. The performance changes of this version do not
  change the products.
  `previous_behavior: true` (or `HGRS_PREVIOUS_BEHAVIOR=1`) gives the 1.1.2 water vapor retrieval
  back (same TCWV), but not exactly the 1.1.2 aerosol retrieval: Rrs then differs by -2.3 % at
  418 nm and -0.7 to -1.3 % from 459 to 577 nm.
- AOT smoothing set in the configuration (`aerosol: smoothing_method`, `smoothing_window`,
  `smoothing_footprint`): `weighted_local` by default, mean of the AOT over 3 x 3 mega-pixels weighted
  by their number of water pixels (instead of no weighted smoothing), then NaN-mean gap filling over
  3 x 3 mega-pixels; `none` (raw AOT) and `nanmean_local` are the other methods.
  Mega-pixels without a retrieved AOT get a null weight.
- Water vapor and aerosol solvers: parameters in the configuration (first guess, fit wavelengths,
  iterations, tolerance), weighted local smoothing of the water vapor, spatially varying AOT for the
  bidirectional transmittance (A. Lavard-Meyer).
- The configuration files moved to `hgrs/config/` (`default_config.yml`, `old_config.yml`).
- The NaN-mean gap filling of the AOT uses two box filters instead of a Python call per pixel.
- The final transmittance computed for the smoothed AOT of each mega-pixel
  (`bidirectional_transmittance_global: false`) is interpolated and applied by blocks of 256 x 256
  pixels, in place: no full-resolution transmittance cube (peak memory 20.3 -> 13.8 GB on an EnMAP
  image, same product).
- `Process.execute` no longer computes the AOT and atmospheric rasters twice (once in `execute`,
  once in `Aerosol.get_atmo_parameters`).
- Each module logs through its own logger (`logging.getLogger(__name__)`); importing `hgrs` no
  longer changes the level of the root logger. The `hgrs_prisma` and `hgrs_enmap` commands still
  print INFO messages; in scripts and notebooks, call `logging.basicConfig(level=logging.INFO)` to
  see them.
- The LUT path of the packaged configuration files is not set (`data_root: null`): the default folder
  of the LUTs is used.
- `pyyaml`, `matplotlib`, `cartopy`, `docopt` and `dask` are declared dependencies; `importlib_resources`
  is no longer needed.
- The development notebooks are no longer tracked.
- Drivers in the `hgrs.drivers` subpackage, one module per sensor (`EnmapDriver`, `PrismaDriver`,
  `EmitDriver`, `HyperionDriver`, `TanagerDriver`) with the common interface `L1Driver.read(img_path, ...)`;
  `Process.execute` selects the driver by the `sensor` name (`hgrs.drivers.get_driver`), and each driver
  declares its required geometry arguments and the chunking of its L2A product. `sensor='prisma'` can be
  given explicitly. An unreadable EnMAP image raises an error instead of stopping silently.
- `hgrs.Driver` is deprecated (kept with a `DeprecationWarning`); `EmitDriver` and `HyperionDriver`
  take the image path in `read` (or, optionally, in the constructor).
- The command-line tools are in one module, `hgrs.cli` (one function per sensor), instead of
  `hgrs.run_enmap`, `hgrs.run_prisma`...: the `hgrs_*` commands and their options are unchanged,
  `python -m hgrs.run_*` is replaced by the commands.

### Fixed
- `filter2d`: the window wrapped around the edges of the image and the mean was computed in place,
  from already smoothed neighbours.
- Replace the deprecated `pkg_resources.resource_filename` by `importlib.resources` and `np.trapz`
  by `np.trapezoid` (NumPy 2).
- Log messages of `run_enmap` and `run_prisma` refer to `hgrs_process` instead of `grs_process`.
- `run_enmap` docstring: EnMAP L1C input instead of PRISMA L1.
- Read the Docs build configuration.
- The wheel contains only the `hgrs` package: the 1.1.2 wheel also installed top-level `docs`,
  `test`, `utils` and `postprocessing` packages.

## [1.1.2] - 2026-06-23

### Changed
- Smoothing windows set to [1, 1] for `filter2D` (the smoothing still needs improvement).
- Command-line interface of `hgrs_prisma` updated.

## [1.1.1] - 2026-04-22

### Changed
- Packaging with `pyproject.toml` and `requirements.txt` (replaces `setup.py`); LUTs and
  auxiliary data moved to `hgrs/data`.
- The EnMAP driver is merged into `driver.py`.
- Comparison of Gaussian and super-Gaussian functions for the spectral integration.

### Removed
- Obsolete scripts (`driverEnMAP.py`, `reader_prisma_working.py`).

## [1.1.0] - 2026-03-25

### Added
- New retrieval of the aerosol optical thickness based on the non-negativity of the retrieved
  reflectance (new inversion scheme).

### Removed
- `hgrs/run.py`.

## [1.0.7] - 2025-11-13

### Added
- Option to remove the bad EnMAP bands.

### Fixed
- Earth-Sun distance correction.

### Changed
- Refactoring of the drivers, of the processing kernel and of the utilities.

## [1.0.6] - 2025-04-25

### Added
- Reprojection (geoprojection) feature.
- EnMAP driver, and visualization of EnMAP and PRISMA L1C images.

### Changed
- Links to the LUTs (`hgrsdata`); output export and examples.

## [1.0.5] - 2023-06-02

### Added
- Match-up analysis and investigation of the solar irradiance reference model.
- Operational processing chain (2024-11) and examples.

### Changed
- Output export; `.gitignore` and clean up of the cached files (2025-04).

## [1.0.4] - 2023-04-04

### Added
- Smoothing of the AOT retrieval based on the uncertainty given by the optimal estimation.

### Changed
- Reorganization of the modules.

## [1.0.3] - 2023-03-28

### Added
- First version of the processing kernel (from GRS).
- Gaseous transmittance correction.

## [1.0.2] - 2023-03-03

### Changed
- Updated solar irradiance spectrum and spectral convolution for the conversion to reflectance.

## [1.0.1]

### Changed
- Rasters oriented North (pending a proper georeference).

## [1.0.0] - 2023-01-18

### Added
- Opening and visualization of PRISMA images, interactive tools, example and installation
  instructions.
