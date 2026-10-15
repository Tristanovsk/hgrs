# Changelog

All notable changes to hGRS are listed here, most recent first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). Dates are those of the commits that set the version.

## [Unreleased]

### Added
- User configuration file `~/.config/hgrs/config.yml` (or `$XDG_CONFIG_HOME/hgrs/config.yml`),
  read at run time on top of the packaged defaults of `hgrs/config/default_config.yml`; the `HGRS_CONFIG` and
  `HGRS_DATA_ROOT` environment variables select another file or the LUT folder.
- `hgrs_config` command to print the configuration in use.
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
- `Process.execute` no longer computes the AOT and atmospheric rasters twice (once in `execute`,
  once in `Aerosol.get_atmo_parameters`).
- Each module logs through its own logger (`logging.getLogger(__name__)`); importing `hgrs` no
  longer changes the level of the root logger. The `hgrs_prisma` and `hgrs_enmap` commands still
  print INFO messages; in scripts and notebooks, call `logging.basicConfig(level=logging.INFO)` to
  see them.
- The LUT path of the packaged configuration files is a placeholder (`path/to/lut`).
- `pyyaml`, `matplotlib`, `cartopy`, `docopt` and `dask` are declared dependencies; `importlib_resources`
  is no longer needed.
- The development notebooks are no longer tracked.

### Fixed
- `filter2d`: the window wrapped around the edges of the image and the mean was computed in place,
  from already smoothed neighbours.
- Replace the deprecated `pkg_resources.resource_filename` by `importlib.resources` and `np.trapz`
  by `np.trapezoid` (NumPy 2).
- Log messages of `run_enmap` and `run_prisma` refer to `hgrs_process` instead of `grs_process`.
- `run_enmap` docstring: EnMAP L1C input instead of PRISMA L1.
- Read the Docs build configuration.

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
