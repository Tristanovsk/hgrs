# Changelog

All notable changes to hGRS are listed here, most recent first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). Dates are those of the commits that set the version.

## [Unreleased]

### Changed
- Lower memory use, same products (bit for bit): the TOA reflectance cube is read once into a single array, and
  masked (water, clouds, bands of strong gaseous absorption) and corrected for the other gases in place; the
  coarse raster is averaged by strips of mega-pixels; GDAL block cache of 256 MB for EnMAP (`GDAL_CACHEMAX`, kept
  if set). Peak memory for a full EnMAP scene: 14.9 to 10.1 GB.

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
