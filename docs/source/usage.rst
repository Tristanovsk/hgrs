Usage
=====

hGRS needs, for each image:

* one Level-1 product: an EnMAP L1C directory, or a PRISMA L1 file (``PRS_L1_STD_OFFL_*.he5``)
  together with the corresponding L2C file, from which the viewing angles are read;
* a CAMS atmospheric composition file (NetCDF) covering the date and the area, with the surface
  pressure (``sp``), the columns of ozone (``gtco3``), nitrogen dioxide (``tcno2``) and methane
  (``tc_ch4``), and the AOD at 469, 550, 670, 865 and 1240 nm.

Command line
------------

EnMAP:

.. literalinclude:: ../../hgrs/run_enmap.py
   :language: text
   :start-at: Usage:
   :end-before: '''

PRISMA:

.. literalinclude:: ../../hgrs/run_prisma.py
   :language: text
   :start-at: Usage:
   :end-before: '''

Python
------

.. code-block:: python

   import hgrs

   process = hgrs.Process()

   # EnMAP: path of the L1C directory
   process.execute('ENMAP01-____L1C-DT0000001234_20240701T103000Z_001_V010400_20240702T000000Z',
                   'cams_2024-07.nc')
   # PRISMA: [L1, L2C] files
   # process.execute(['PRS_L1_STD_OFFL_20210906103712_20210906103717_0001.he5',
   #                  'PRS_L2C_STD_20210906103712_20210906103717_0001.he5'], 'cams_2021-09.nc')

   process.l2_prod            # xarray.Dataset of the L2A product
   process.write_output('ENMAP01_L2A_hgrs.nc')

The individual steps can also be run one by one with the classes of :py:mod:`hgrs.hgrs_kernel` (see
the :doc:`tutorials <tutorials/process_image>` and the :doc:`api/hgrs_kernel` overview).

Output product
--------------

The L2A product is a NetCDF file, restricted to 400-1150 nm, with the variables:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Variable
     - Grid
     - Unit
     - Description
   * - ``Rrs``
     - wl, y, x
     - sr\ :sup:`-1`
     - remote-sensing reflectance :math:`R_{rs}(\lambda)`
   * - ``brdfg_full``
     - y, x
     - \-
     - sunglint amplitude :math:`B` at full resolution
   * - ``aot_ref``
     - yc, xc
     - \-
     - aerosol optical thickness at 550 nm (coarse grid)
   * - ``aot_ref_smoothed``
     - yc, xc
     - \-
     - smoothed and gap-filled ``aot_ref``, used for the correction
   * - ``aot_ref_std``, ``brdfg_std``
     - yc, xc
     - \-
     - quality indicators of the optimization (final cost, squared norm of the gradient)
   * - ``brdfg``
     - yc, xc
     - \-
     - sunglint amplitude :math:`B` retrieved with the aerosols (coarse grid)
   * - ``tcwv``, ``tcwv_std``
     - yc, xc
     - kg m\ :sup:`-2`
     - total column water vapor fitted on the image and its uncertainty
   * - ``water_pix_prop``
     - yc, xc
     - \-
     - proportion of water pixels in each mega-pixel of the coarse grid
   * - ``pressure``, ``to3c``, ``tno2c``
     - scalar
     - hPa, CAMS units
     - surface pressure, ozone and nitrogen dioxide columns from CAMS

The processing parameters (wavelength sets, thresholds, LUT files) are stored in the global
attributes.
