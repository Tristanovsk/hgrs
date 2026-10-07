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

Input data
----------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.driver.Driver
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

Spectral convolution
--------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   hgrs.hgrs_kernel.Spectral

.. autosummary::
   :toctree: generated
   :nosignatures:

   hgrs.hgrs_kernel.gaussian
   hgrs.hgrs_kernel.Gamma2sigma
   hgrs.hgrs_kernel.super_gaussian
   hgrs.hgrs_kernel.super_gaussian_fwhm2sigma

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
