

import os, copy
import glob

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
plt.ioff()
import colorcet as cc

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr
import datetime as dt

import hgrs.driver as driver
import hgrs

opj = os.path.join



l1cdir = '/sat_data/satellite/acix-iii/AERONET-OC'
workdir='/media/harmel/TOSHIBA EXT/acix-iii'
sites = os.listdir(workdir)

wl_rgb=[30, 20, 6]
gamma=0.5
brightness_factor=1
#workdir = '/sat_data/satellite/acix-iii'
l1cdir = '/sat_data/satellite/acix-iii'

sites= ['Wendtorf', 'Varese', 'Venice_Lagoon', 'Geneve', 'Garda', 'Trasimeno']
sites = next(os.walk(workdir))[1]
sites.remove('v0')

for site in sites:
    workdir_ =opj(workdir,site)

    for img_path in glob.glob(opj(workdir_,'*L2A*.nc')):



        img = xr.open_dataset(img_path)
        if img.Rrs.encoding['zlib']:
            continue
        print(img_path)
        complevel = 5
        encoding = {
            'Rrs': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True,
                    "complevel": complevel},
            'aot_ref_full': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
                             "complevel": complevel},
            'aot_ref': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
                        "complevel": complevel},
            'aot_ref_std': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
                            "complevel": complevel},
            'brdfg_full': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768,
                           "zlib": True, "complevel": complevel},
            'brdfg': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True,
                      "complevel": complevel},
            'brdfg_std': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768,
                          "zlib": True, "complevel": complevel},
            'tcwv_full': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True,
                          "complevel": complevel},
            'tcwv': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True, "complevel": complevel},
            'tcwv_std': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True,
                         "complevel": complevel}}
        os.remove(img_path)
        img.to_netcdf(img_path, encoding=encoding,mode='w')
        img.close()