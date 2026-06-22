Usage
=====

Installation
------------

These instructions will get you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on how to deploy the project on a live system.

::

pip install hgrs


If you have no GDAL implementation yet, please une anaconda environment and install GDAL before hGRS:

::

conda install -c conda-forge gdal


Installation from hGRS repository (enables use of the notebook and visualization tools)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. First clone the repository:

.. code::

git clone https://github.com/Tristanovsk/hgrs.git


2.a. You may duplicate the conda environment (optional)

.. code::

conda env create -f environment.yml


2.b. If you do not duplicate the environment please install GDAL with conda

.. code::

conda install -c conda-forge gdal


3. Install GRS

.. code::

conda activate grssuite

conda install -c pyviz holoviews panel

pip install .


4. Install and configure JupyterLab and Holoviews

.. code::

pip install holoviews datashader cartopy

conda install -c conda-forge jupyterlab




You are done.
