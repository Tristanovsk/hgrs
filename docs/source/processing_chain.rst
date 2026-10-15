Processing Chain Description
=============================

This page documents hGRS following the
`Processing Chain Documentation Template <https://processing-chain-guidelines.readthedocs.io/en/latest/Data_processing_chain_template/>`_.

Process description
--------------------

Description
~~~~~~~~~~~

hGRS (hyperspectral Glint Removal for Sentinel-2-like sensors) is an atmospheric and sunglint
correction processor for hyperspectral satellite images acquired over aquatic environments. Given one
Level-1 (top-of-atmosphere) image, it corrects successively for gaseous absorption, water vapor,
aerosol scattering and sunglint, to retrieve the remote-sensing reflectance :math:`R_{rs}(\lambda)`
at the surface level. The scientific description is given in :doc:`methods` and in the
:doc:`api/hgrs_kernel` overview.

The processor is invoked through one command-line executable per sensor, which wraps the core
:py:class:`hgrs.hgrs_process.Process` class:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Command
     - Entry point
     - Sensor
   * - ``hgrs_enmap``
     - :py:func:`hgrs.run_enmap.main`
     - EnMAP L1C
   * - ``hgrs_prisma``
     - :py:func:`hgrs.run_prisma.main`
     - PRISMA L1 (+ L2C for the angles)
   * - ``hgrs_emit``
     - :py:func:`hgrs.run_emit.main`
     - EMIT L1B radiance
   * - ``hgrs_hyperion``
     - :py:func:`hgrs.run_hyperion.main`
     - EO-1 Hyperion L1R
   * - ``hgrs_tanager``
     - :py:func:`hgrs.run_tanager.main`
     - Tanager basic radiance

The ``hgrs_config`` command prints the configuration in use and checks the LUTs;
``hgrs_config --init`` creates the user configuration file (see :doc:`configuration`).

Application Domain (Granule)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One processing run (one granule) corresponds to **one Level-1 image for one acquisition**:

- one EnMAP L1C product directory, or
- one PRISMA L1 ``.he5`` file, with the L2C file of the same acquisition, or
- one EMIT L1B radiance granule, one Hyperion L1R scene or one Tanager basic radiance file.

The output granule is a single Level-2A hGRS NetCDF file covering the footprint of the input image,
either reprojected on a map grid (default) or on the native pixel grid with the 2D longitude and
latitude arrays (``--no-geoprojection``).

Scheduling and Triggers
~~~~~~~~~~~~~~~~~~~~~~~~

hGRS has no built-in scheduler; it is triggered externally, image by image:

- **Interactively / single image**: manual call to ``hgrs_enmap <l1_path> --cams_file <file> ...``
  or another sensor command (see :doc:`usage`), or with the Python API
  (:py:meth:`Process.execute <hgrs.hgrs_process.Process.execute>`, see the
  :doc:`tutorials <tutorials/process_image>`).
- **Batch**: a list of commands, one per image, is generated (Level-1 images of the sites and CAMS file
  covering each acquisition) and run from a shell script or with GNU ``parallel``, see the
  :doc:`batch processing tutorial <tutorials/batch_processing>`. ``--no_clobber`` skips the images
  already processed (``hgrs_enmap``, ``hgrs_prisma``), so that the same list can be run again after
  new acquisitions.

There is no periodicity of its own; scheduling (e.g. processing new Level-1 acquisitions) is delegated
to the calling scripts or job scheduler.

Dataflow
--------

.. mermaid:: _diagrams/dataflow.mmd

Inputs
------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Data Type Name
     - Cardinality
     - Selection Criteria
   * - Level-1 image (``<l1_path>``)
     - 1..1 (mandatory)
     - One EnMAP L1C directory, PRISMA L1 ``.he5`` file, EMIT L1B ``.nc`` granule, Hyperion scene
       (``.L1R`` file or directory containing one) or Tanager ``.h5`` file, passed as the positional CLI
       argument.
   * - PRISMA L2C file (``--l2_path``)
     - 0..1 (PRISMA only)
     - L2C product of the same acquisition, from which the viewing angles and the land mask are read;
       by default, searched next to the L1 file (``L1_STD_OFFL`` replaced by ``L2C_STD``).
   * - CAMS file (``--cams_file``)
     - 1..1 (mandatory)
     - Copernicus Atmosphere Monitoring Service NetCDF file (forecast or reanalysis) covering the
       acquisition date and area: surface pressure, O\ :sub:`3`, NO\ :sub:`2`, CH\ :sub:`4` columns and
       aerosol optical depth, used to select the aerosol model and to set the gaseous absorption and the
       first guess and bounds of the aerosol retrieval.
   * - Scene geometry (``--sun_azimuth``, ``--sun_elevation``, ...)
     - 1..1 (EMIT, Hyperion)
     - Solar and viewing angles of the acquisition, not read from the EMIT and Hyperion files.
   * - Look-up tables
     - 1..1 (mandatory)
     - Radiative transfer LUTs (``toa_lut``, ``trans_lut``) in the ``data_root`` folder of the
       configuration (see :doc:`configuration`); the gaseous absorption LUT and the solar irradiance
       spectra are shipped with the package.
   * - User configuration file
     - 0..1 (optional)
     - ``~/.config/hgrs/config.yml`` (or ``HGRS_CONFIG``), overriding the packaged defaults
       (``hgrs/config/default_config.yml``): LUT folder, masks thresholds, retrieval and smoothing
       parameters.

Outputs
-------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Data Type Name
     - Cardinality
     - Description
   * - Level-2A hGRS product (NetCDF)
     - 1..1
     - Remote-sensing reflectance and retrieved atmospheric and sunglint parameters, one NetCDF file per
       image (see `Output product format`_ and `Data Types`_ for naming).
   * - Log messages
     - 1..1
     - Written to the standard error stream (see `Log Format`_); no log file is written by hGRS.

Output product format
~~~~~~~~~~~~~~~~~~~~~

The product is written by :py:meth:`Process.write_output <hgrs.hgrs_process.Process.write_output>`
into a single NetCDF file (an existing file of the same name is overwritten):

* **Spectral range**: 400-1150 nm; the bands of strong gaseous absorption are removed.
* **Grids**: full resolution (``x``, ``y``) for ``Rrs`` and the sunglint amplitude ``brdfg_full``,
  coarse grid of the retrieval mega-pixels (``xc``, ``yc``) for the aerosol, sunglint and water vapor
  retrievals and their uncertainties; see :doc:`usage` for the list of variables.
* **Masks**: ``water_mask`` and ``cloud_mask`` (stored as ``uint8``) and, for PRISMA, ``land_mask``
  (True for the pixels kept for the retrievals according to the land cover classes of the L2C product).
* **Packing**: the continuous variables are stored as scaled ``int16`` (e.g. ``Rrs``: scale 1e-5,
  offset 0.2), zlib compression level 5; they are unpacked automatically when read.
* **Metadata**: global attributes with the processing and acquisition dates, hGRS version, aerosol
  model, scene parameters (pressure, gas columns) and ``geoprojection`` (``reprojected`` or
  ``native``).

Read the product back with xarray:

.. code-block:: python

   import xarray as xr

   l2a = xr.open_dataset('<name>.nc')
   l2a.Rrs.sel(wl=560, method='nearest').plot()

Return Codes
------------

hGRS is a Python CLI (``docopt``-based, ``argparse`` for ``hgrs_emit``); errors are not caught by the
commands, so they are reported through the exit code of the Python interpreter:

- ``0``: normal process exit, the product has been written. Also returned by ``hgrs_enmap`` and
  ``hgrs_prisma`` when the image is skipped because the output file already exists and
  ``--no_clobber`` is set (the message ``File ... already processed; skip!`` is printed).
- ``1``: processing error (unreadable input, missing CAMS or LUT file, retrieval failure...): the
  exception and its traceback are printed on the standard error stream and no product is written. Also
  returned for invalid arguments rejected by ``docopt``.
- ``2``: invalid arguments rejected by ``argparse`` (``hgrs_emit``).

Log Format
----------

Each module logs through its own logger (``logging.getLogger(__name__)``, e.g. ``hgrs.hgrs_process``).
``hgrs_enmap`` and ``hgrs_prisma`` configure the root logger with ``logging.basicConfig(level=INFO)``,
so the messages are written to the standard error stream with the default format of the
:py:mod:`logging` module:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``levelname``
     - Classification level: ``DEBUG``, ``INFO``, ``WARNING`` or ``ERROR``.
   * - ``name``
     - Logger name, i.e. the module emitting the record (e.g. ``hgrs.hgrs_process``).
   * - ``message``
     - Free-text message body (processing step, retrieved aerosol model...).

Format string: ``"%(levelname)s:%(name)s:%(message)s"``.

The commands also print the parsed arguments at start-up. ``hgrs_emit``, ``hgrs_hyperion`` and
``hgrs_tanager`` do not configure logging, so only the warnings and errors are shown. In scripts and
notebooks, call ``logging.basicConfig(level=logging.INFO)`` (or any handler and format) to see the
messages; to keep a log file, redirect the standard error stream (``hgrs_enmap ... 2> run.log``).

Required Resources
-------------------

Actual needs depend on the image size and on the fraction of water pixels. Indicative values:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Resource Type
     - Quantity
     - Source / notes
   * - CPU
     - all the cores of the node
     - the water vapor and aerosol retrievals are run in parallel on blocks of mega-pixels
       (:py:class:`multiprocessing.Pool` with one process per core); run several images at a time
       with care (see the :doc:`batch processing tutorial <tutorials/batch_processing>`)
   * - RAM
     - the Level-1 image cube in memory, plus the product
     - the full-resolution correction is computed by blocks of 256 x 256 pixels
   * - Execution time
     - not measured systematically
     - dominated by the pixel-wise water vapor and aerosol retrievals on the mega-pixels with water
   * - Disk — input
     - one Level-1 image (e.g. about 450 MB for an EnMAP L1C product), the CAMS NetCDF file and the
       LUTs (about 850 MB)
     - the LUTs are shared by all the runs
   * - Disk — output
     - one Level-2A NetCDF file per image, from a few MB to a few hundred MB
     - written under ``--odir``; the size grows with the number of water pixels (NaN elsewhere,
       compressed)

Data Types
----------

.. list-table::
   :header-rows: 1
   :widths: 18 32 15 35

   * - Data Name
     - Description
     - Granule
     - Nomenclature
   * - EnMAP L1C product
     - Top-of-atmosphere radiance, orthorectified
     - one scene, one date
     - ``ENMAP01-____L1C-DT<datatake>_<YYYYMMDDTHHMMSSZ>_<tile>_V<version>_<YYYYMMDDTHHMMSSZ>``
       (directory)
   * - PRISMA L1 / L2C products
     - Top-of-atmosphere radiance / surface reflectance with the viewing angles
     - one scene, one date
     - ``PRS_L1_STD_OFFL_<start>_<stop>_<id>.he5`` / ``PRS_L2C_STD_<start>_<stop>_<id>.he5``
   * - EMIT L1B product
     - Top-of-atmosphere radiance
     - one granule, one date
     - ``EMIT_L1B_RAD_<version>_<YYYYMMDDTHHMMSS>_<orbit>_<scene>.nc``
   * - Hyperion L1R product
     - Top-of-atmosphere radiance
     - one scene, one date
     - ``EO1H<path><row><YYYYDDD>...L1R``
   * - Tanager product
     - Top-of-atmosphere radiance
     - one scene, one date
     - ``<YYYYMMDD>_<HHMMSS>_<id>_basic_radiance_hdf5.h5``
   * - CAMS file
     - Copernicus Atmosphere Monitoring Service atmospheric composition forecast (or reanalysis)
     - regional/global grid, one day or one month
     - e.g. ``YYYY-MM-DD-cams-global-atmospheric-composition-forecasts.nc`` or
       ``cams_forecast_YYYY-MM.nc``
   * - Look-up tables
     - Radiative transfer simulations of the TOA reflectance and of the transmittances
     - static
     - ``toa_lut_opac_wind_up_v3.nc``, ``transmittance_lut_opac_wind_v3.nc`` (configuration ``path``)
   * - Level-2A hGRS output
     - NetCDF file of the remote-sensing reflectance product
     - one scene, one date
     - ``<odir>/<name>`` with ``<name>`` given by ``-o``, or by default: EnMAP, input name with
       ``L1C`` replaced by ``--levname`` (default ``L2A_hgrs``) + ``_V<hGRS version>.nc``; PRISMA,
       input name with ``L1_STD_OFFL`` replaced by ``--levname`` and ``.he5`` by
       ``_V<hGRS version>.nc``; EMIT and Hyperion, ``<scene>_L2A_hgrs_V<hGRS version>.nc``; Tanager,
       ``basic_radiance_hdf5.h5`` replaced by ``L2A_hgrs_V<hGRS version>.nc``.
