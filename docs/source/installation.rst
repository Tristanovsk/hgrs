Installation
============

Look-up tables
--------------

The radiative transfer look-up tables (LUT) are not part of the package: ``toa_lut_opac_wind_v3.nc``
(835 MB) and ``transmittance_lut_opac_wind_v3.nc`` (0.6 MB), shared with GRS and published on Zenodo
(`doi:10.5281/zenodo.23394202 <https://doi.org/10.5281/zenodo.23394202>`__). Once the package is
installed, download them into the default folder of hGRS, ``~/.local/share/hgrs/lut`` on Linux (no
configuration needed, see :doc:`configuration`):

.. code-block:: bash

   hgrs_config --download

or download them from the Zenodo page into a directory of your choice (``your_LUT_PATH`` below). The
former name of the TOA LUT, ``toa_lut_opac_wind_up_v3.nc`` (the same LUT, uncompressed), is still found
in the LUT folder.

Python environment
------------------

GDAL and xESMF (used for the geoprojection of PRISMA images) are easier to install with conda. From
the root of the repository:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate hgrs

or, in an existing environment:

.. code-block:: bash

   conda install -c conda-forge gdal xesmf

Install the package
-------------------

From the root of the repository:

.. code-block:: bash

   pip install .

Then check the command-line tools:

.. code-block:: bash

   hgrs_enmap -v
   hgrs_prisma -v

Configuration
-------------

If the LUTs are not in the default folder, write their folder in your user configuration file:

.. code-block:: bash

   hgrs_config --init your_LUT_PATH

Then check the configuration and the LUTs:

.. code-block:: bash

   hgrs_config

See :doc:`configuration` for all the settings.

For the visualization tools of the notebooks:

.. code-block:: bash

   conda install -c conda-forge jupyterlab holoviews hvplot panel datashader bokeh cartopy

Build this documentation
------------------------

.. code-block:: bash

   pip install ".[docs]"
   sphinx-build -b html docs/source docs/build/html

The example notebooks are not executed: they are rendered with the outputs saved in them.
