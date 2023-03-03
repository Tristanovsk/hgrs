
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
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import colorcet as cc

import prismapy.driver as driver
from prismapy import metadata

opj = os.path.join

aux = metadata()
workdir = '/sat_data/satellite/acix-iii/Garda'
l1c = 'PRS_L1_STD_OFFL_20210721102700_20210721102705_0001.he5'
l2c = 'PRS_L2C_STD_20210721102700_20210721102705_0001.he5'

l2c_path = opj(workdir,l2c)
dc = driver.read_L2C_data(l2c_path)

fig = dc.rho.isel(wl=[1,10,20,30,40,50,90,130,200]).plot.imshow(col='wl',col_wrap=3,robust=True,cmap=cc.cm.bky)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')

plt.savefig('fig/test_L2C_Garda.png',dpi=150)
