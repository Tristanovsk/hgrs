.. _api:

API reference
=============

The main classes are importable from the package itself:

.. code-block:: python

   import hgrs

   process = hgrs.Process()
   process.execute('ENMAP01-____L1C-..._V010400_...', 'cams.nc')
   process.write_output('ENMAP01_L2A_hgrs.nc')

Processing chain
----------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.hgrs_process.Process

.. autosummary::
   :toctree: generated
   :nosignatures:

   hgrs.run_enmap.main
   hgrs.run_prisma.main
   hgrs.run_emit.main
   hgrs.run_hyperion.main
   hgrs.run_tanager.main

L1 drivers
----------

.. automodule:: hgrs.drivers
   :no-members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   hgrs.drivers.get_driver

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.drivers.base.L1Driver
   hgrs.drivers.EnmapDriver
   hgrs.drivers.PrismaDriver
   hgrs.drivers.EmitDriver
   hgrs.drivers.HyperionDriver
   hgrs.drivers.TanagerDriver

Input data
----------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.auxdata.AuxData
   hgrs.auxdata.SolarIrradiance

Atmospheric and sunglint correction
-----------------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.hgrs_kernel.Product
   hgrs.hgrs_kernel.Algo
   hgrs.hgrs_kernel.WaterVapor
   hgrs.hgrs_kernel.Aerosol
   hgrs.hgrs_kernel.Solver

Configuration
-------------

Settings read by :py:func:`hgrs.config.get_app_config` (see :doc:`configuration`); the
switch :py:data:`hgrs.config.behavior.PREVIOUS_BEHAVIOR` selects the former parameters and algorithms.

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.config.AppConfig
   hgrs.config.AtmosphericCorrectionConfig
   hgrs.config.ProductCorrectionConfig
   hgrs.config.WaterVaporConfig
   hgrs.config.AerosolCorrectionConfig
   hgrs.config.SceneMetadata
   hgrs.config.SensorDescription

.. autosummary::
   :toctree: generated
   :nosignatures:

   hgrs.config.get_app_config
   hgrs.config.loader.load_config
   hgrs.config.loader.default_data_root
   hgrs.config.loader.init_user_config
   hgrs.config.loader.main

Spectral convolution
--------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.spectral_sensitivity.Gaussian
   hgrs.spectral_sensitivity.SuperGaussian

Former implementation, kept in :py:mod:`hgrs.legacy.spectral`:

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.legacy.spectral.Spectral

.. autosummary::
   :toctree: generated
   :nosignatures:

   hgrs.legacy.spectral.gaussian
   hgrs.legacy.spectral.Gamma2sigma
   hgrs.legacy.spectral.super_gaussian
   hgrs.legacy.spectral.super_gaussian_fwhm2sigma

Utilities
---------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.utils.Misc
   hgrs.utils.Reproj

Module overview
---------------

Physical model of the kernel, with the notation and the equations.

.. toctree::
   :maxdepth: 1

   api/hgrs_kernel
