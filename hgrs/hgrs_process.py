
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

import hgrs.driver as driver
import hgrs

opj = os.path.join

class hgrs():
    def __init__(self):
        # parameters
        self.wl_water_vapor = slice(800, 1300)
        self.wl_glint = slice(2100, 2200)
        self.wl_atmo = slice(950, 2250)
        self.wl_to_remove = [(935, 967), (1105, 1170), (1320, 1490), (1778, 2033)]
        self.xcoarsen = 20
        self.ycoarsen = 20
        self.block_size = 2
        self.dirdata =
        self.abs_gas_file = '/DATA/git/vrtc/libradtran_tbx/output/lut_abs_opt_thickness_normalized.nc'
        self.lut_file = '/media/harmel/vol1/work/git/vrtc/RTxploitation/study_cases/aerosol/lut/opac_osoaa_lut_v2.nc'
        self.pressure_rot_ref = 1013.25

        # TODO get them from CAMS
        self.to3c = 6.5e-3
        self.tno2c = 3e-6
        self.tch4c = 1e-2
        self.psl = 1013
        self.coef_abs_scat = 0.3

        # In[5]:

        l1c_path = opj(workdir, l1c)
        l2c_path = opj(workdir, l2c)

        dc_l1c = driver.read_L1C_data(l1c_path, reflectance_unit=True, drop_vars=True)
        dc_l2c = driver.read_L2C_data(l2c_path)
