

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



l1cdir = '/data/satellite/acix-iii/AERONET-OC'
workdir='/data/acix-iii'
sites = next(os.walk(workdir))[1]
sites.remove('v0')
wl_rgb=[30, 20, 6]
gamma=0.5
brightness_factor=1
#workdir = '/sat_data/satellite/acix-iii'

l1cdir = '/data/satellite/acix-iii'
sites = ['Wendtorf', 'Varese', 'Venice_Lagoon', 'Geneve', 'Garda', 'Trasimeno']
odir='/data/satellite/acix-iii/fig/L2A'
#
# l1cdir = '/data/data/satellite/prisma/zoffoli'
# workdir='/data/data/satellite/prisma/zoffoli/L2A'
# odir = '/data/data/satellite/prisma/zoffoli/L2A'
# sites=['']

l1cdir = '/data/acix-iii/Second_Batch/L1'#/AERONET-OC'
workdir='/data/acix-iii/Second_Batch/L2A'
sites = next(os.walk(workdir))[1]
odir='/data/acix-iii/Second_Batch/fig'


l1cdir = '/data/satellite/prisma/zoffoli/L1'#/AERONET-OC'
workdir='/data/satellite/prisma/zoffoli/L2A'
sites = ['']
odir='/data/satellite/prisma/zoffoli/fig'
pattern = 'L2A_hGRSv2'

l1cdir = '/data/satellite/prisma/berre/L1C'#/AERONET-OC'
workdir='/data/satellite/prisma/berre/L2A'
sites = ['']
odir='/data/satellite/prisma/berre/fig'
pattern = 'L2A_hGRS'

l1cdir = '/data/acix-iii/Third_Batch/L1'#/AERONET-OC'
workdir='/data/acix-iii/Third_Batch/L2A'
sites = [''] #next(os.walk(workdir))[1]
odir='/data/acix-iii/Third_Batch/fig'
pattern = 'L2A_hGRSv2'

l1cdir = '/data/satellite/prisma/touria'#/AERONET-OC'
workdir='/data/satellite/prisma/touria'
sites = ['bassas','glorieuse','mayotte'] #next(os.walk(workdir))[1]
odir='/data/satellite/prisma/touria/fig'
pattern = 'L2A_hgrs'


for site in sites:
    workdir_ =opj(workdir,site,'L2A')
    if not os.path.isdir(workdir_):
        continue
    l1cdir_ = opj(l1cdir,site)
    for img_path in glob.glob(opj(workdir_,'*'+pattern+'*.nc')):
        basename = os.path.basename(img_path)

        figname = opj(odir, site + '_' + basename.replace('.nc', '.png'))
        print(img_path)
        if os.path.exists(figname):
            pass
            #continue

        l1c_path = opj(l1cdir_,'L1',basename.replace('.nc','.he5').replace( pattern,'L1_STD_OFFL'))
        dc_l1c = driver.read_L1C_data(l1c_path, reflectance_unit=True, drop_vars=True)

        img = xr.open_dataset(img_path)
        date= dt.datetime.strptime(img.acquisition_date,'%Y-%m-%dT%H:%M:%S.%f')

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.ravel()
        ax = (dc_l1c.Rtoa.isel(wl=wl_rgb) ** gamma * brightness_factor).plot.imshow(rgb='wl', robust=True,ax=axs[0])
        ax.axes.set_aspect('equal')

        rgb = img.Rrs.isel(wl=wl_rgb)
        adj = xr.DataArray([1.2, 1, 1], coords={"wl": rgb.wl})
        ax = (rgb * adj).plot.imshow(rgb='wl', robust=True, ax=axs[1])
        ax.axes.set_aspect('equal')

        ax=img['aot_ref_full'].plot.imshow(cmap=plt.cm.Spectral_r, robust=True, vmin=0, ax=axs[2],
                                        cbar_kwargs={'shrink': 0.7})#, add_colorbar=False)
        ax.axes.set_aspect('equal')
        ax=img['brdfg_full'].plot.imshow(cmap=plt.cm.gray, robust=True, vmin=0, ax=axs[3],
                                      cbar_kwargs={'shrink': 0.7})#, add_colorbar=False)
        ax.axes.set_aspect('equal')
        for i in range(4):
            axs[i].set(xticks=[], yticks=[])
            axs[i].set_ylabel('')
            axs[i].set_xlabel('')
        axs[0].set_title(site+', '+str(date))
        axs[1].set_title("hGRS")
        axs[2].set_title("AOT(550nm)")
        axs[3].set_title("Sunglint")

        plt.tight_layout()

        plt.savefig(figname,dpi=300)
        plt.close()