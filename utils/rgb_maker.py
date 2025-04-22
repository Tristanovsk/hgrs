

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



workdir = '/sat_data/satellite/acix-iii/AERONET-OC'
sites = os.listdir(workdir)

workdir = '/sat_data/satellite/acix-iii'
sites = ['Wendtorf', 'Varese', 'Venice_Lagoon', 'Geneve', 'Garda', 'Trasimeno']

for site in sites:
    workdir_ =opj(workdir,site)


    for l1c_path in glob.glob(opj(workdir_,'*L1_STD_OFFL*.he5')):# 'PRS_L1_STD_OFFL_20210814141750_20210814141755_0001.he5'
        #workdir = '/sat_data/satellite/acix-iii/Garda'
        #l1c = 'PRS_L1_STD_OFFL_20210721102700_20210721102705_0001.he5'
        l1c = os.path.basename(l1c_path)
        figname = opj('/sat_data/satellite/acix-iii/fig/', site + '_' + l1c.replace('.he5', '.png'))
        print(l1c_path)
        if os.path.exists(figname):
            pass
            #continue
        date= dt.datetime.strptime(l1c.split('_')[4],'%Y%m%d%H%M%S')
        l2c = l1c.replace('L1_STD_OFFL','L2C_STD')
        #l1c_path = opj(workdir,l1c)
        l2c_path = opj(workdir,l2c)

        dc_l1c = driver.read_L1C_data(l1c_path,reflectance_unit=True,drop_vars=True)

        # dc_l2c = driver.read_L2C_data(l2c_path)
        # for param in ['sza','vza','raa']:
        #     dc_l1c[param]=dc_l2c[param]

        prod = hgrs.algo(dc_l1c)
        #prod.load_metadata()
        #prod.get_ndwi()
        plt.figure(figsize=(7,7))
        fig = prod.rgb()
        fig.figure.suptitle(site+', '+str(date),fontsize=19)
        fig.figure.tight_layout()
        fig.figure.savefig(figname,dpi=300)
        plt.close()