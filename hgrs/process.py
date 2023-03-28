

import os, copy
import glob

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr

import hgrs.driver as driver
import hgrs

opj = os.path.join


workdir = '/sat_data/satellite/acix-iii/AERONET-OC/bahiablanca'
l1c = 'PRS_L1_STD_OFFL_20210814141750_20210814141755_0001.he5'
#workdir = '/sat_data/satellite/acix-iii/Garda'
#l1c = 'PRS_L1_STD_OFFL_20210721102700_20210721102705_0001.he5'
l2c = l1c.replace('L1_STD_OFFL','L2C_STD')


l1c_path = opj(workdir,l1c)
l2c_path = opj(workdir,l2c)

dc_l1c = driver.read_L1C_data(l1c_path,reflectance_unit=True,drop_vars=True)
dc_l2c = driver.read_L2C_data(l2c_path)

for param in ['sza','vza','raa']:
    dc_l1c[param]=dc_l2c[param]


