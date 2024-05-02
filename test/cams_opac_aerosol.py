import os, copy
import glob

import numpy as np
import pandas as pd
import xarray as xr
#%matplotlib widget
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
import hgrs
opj = os.path.join
plt.ioff()

lut_file = '/DATA/git/satellite_app/hgrs/data/lut/opac_osoaa_lut_v3.nc'
aero_lut = xr.open_dataset(lut_file)
aero_lut['wl']=aero_lut['wl']*1000



cams_files =glob.glob('/DATA/projet/magellium/acix-iii/cams/PRS*.nc')
#'/DATA/projet/magellium/acix-iii/cams/PRS_L1_STD_OFFL_20220930074620_20220930074624_0001_cams.nc'

for cams_file in cams_files:
    basename=os.path.basename(cams_file)
    print(basename)
    cams = xr.open_dataset(cams_file)
    aod = cams[['aod355', 'aod380', 'aod400', 'aod440', 'aod469', 'aod500', 'aod550', 'aod645', 'aod670',
     'aod800', 'aod865','aod1020', 'aod1064', 'aod1240', 'aod1640', 'aod2130']].to_pandas()
    aod.index=aod.index.str.replace('aod','').astype(int)
    cams_aod=aod.to_xarray().rename({'index':'wl'})


    rh='_rh70'
    models = ['ANTA'+rh,'ARCT'+rh,'COAV'+rh, 'COPO'+rh, 'DESE'+rh, 'MACL'+rh, 'MAPO'+rh, 'URBA'+rh]


    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    fig.subplots_adjust(bottom=0.18, top=0.88, left=0.086, right=0.98,
                        hspace=0.22, wspace=0.125)
    aero_lut.aot.sel(model=models,aot_ref=1).plot(ax=axs[0],hue='model')
    (cams_aod/cams.aod550).plot(ax=axs[0],color='black')
    lut_aod=aero_lut.aot.sel(model=models,aot_ref=1).interp(wl=cams_aod.wl)
    rank = np.abs((cams_aod/cams.aod550)-lut_aod).sum('wl')
    axs[1].bar(x=rank.model, height=rank.values)
    plt.xticks(rotation=30, ha='right')
    plt.suptitle(basename)
    plt.savefig(opj('/DATA/projet/magellium/acix-iii/cams_opac',basename.replace('.nc','.png')))