

import os, copy
import glob

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import colorcet as cc

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr



import hgrs.driver as driver
import hgrs

opj = os.path.join


workdir_ = '/sat_data/satellite/acix-iii/'
site='Geneve'
workdir=opj(workdir_,site)
l1c='PRS_L1_STD_OFFL_20210906103712_20210906103717_0001.he5'
l1c_path=opj(workdir,l1c)
#l1c = os.path.basename(l1c_path)

l2c = l1c.replace('L1_STD_OFFL','L2C_STD')
l2c_path = opj(workdir,l2c)

#---------------------------------------
# construct L1C image plus angle rasters
#---------------------------------------
dc_l1c = driver.read_L1C_data(l1c_path,reflectance_unit=True,drop_vars=True)
dc_l2c = driver.read_L2C_data(l2c_path)
for param in ['sza','vza','raa']:
     dc_l1c[param]=dc_l2c[param]
del dc_l2c

#------------------------
# Create hGRS object
#------------------------
prod = hgrs.algo(dc_l1c)
prod.load_metadata()
prod.apply_water_masks()
prod.round_angles()
prod.get_air_mass()
prod.get_coarse_masked_raster()
prod.get_gaseous_transmittance()
prod.other_gas_correction()
wv_retrieval = hgrs.water_vapor(prod)
wv_retrieval.solve()
prod.get_wv_transmittance_raster(wv_retrieval.water_vapor)
prod.water_vapor_correction()


plt.figure(figsize=(10,10))
fig = prod.rgb('Rtoa_masked',raster_name='coarse_masked_raster')
