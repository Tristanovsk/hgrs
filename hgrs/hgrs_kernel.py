
import os, copy
import glob

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr
import rioxarray as rxr

import netCDF4
import h5py
from osgeo import gdal
from pkg_resources import resource_filename

import hgrs.driver as driver
import hgrs

opj = os.path.join

class hgrs():
    def __init__(self,l1c_obj=None):
        # spectral parameters
        self.wl_water_vapor = slice(800, 1300)
        self.wl_glint = slice(2100, 2200)
        self.wl_atmo = slice(950, 2250)
        self.wl_to_remove = [(935, 967), (1105, 1170), (1320, 1490), (1778, 2033)]
        self.wl_green= slice(540, 570)
        self.wl_nir = slice(850, 890)

        # image chunking and coarsening parameters
        self.xcoarsen = 20
        self.ycoarsen = 20
        self.block_size = 2

        # pre-computed auxiliary data
        self.dirdata = resource_filename(__package__, '../data/')
        self.abs_gas_file = opj(self.dirdata,'lut','lut_abs_opt_thickness_normalized.nc')
        self.lut_file = opj(self.dirdata,'lut','opac_osoaa_lut_v2.nc')
        self.pressure_rot_ref = 1013.25

        # atmosphere auxiliary data
        # TODO get them from CAMS
        self.to3c = 6.5e-3
        self.tno2c = 3e-6
        self.tch4c = 1e-2
        self.psl = 1013
        self.coef_abs_scat = 0.3

        # xarray object to be processed
        self.l1c_obj = l1c_obj

    def load_metadata(self):
        self.gas_lut = xr.open_dataset(self.abs_gas_file)
        self.aero_lut = xr.open_dataset(self.lut_file)

    def get_ndwi(self):
        green = self.l1c_obj.sel(wl=self.wl_green).mean(dim='wl')
        nir = self.l1c_obj.sel(wl=self.wl_nir).mean(dim='wl')
        self.ndwi = (green - nir) / (green + nir)



