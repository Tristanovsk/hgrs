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

import datetime as dt
import h5py

import hgrs.driver as driver
import hgrs

opj = os.path.join

info=[]
for case in [0,1]:
    if case == 0:
        workdir_ = '/sat_data/satellite/acix-iii/AERONET-OC'
        sites = ['galataplatform', 'sanmarcoplatform', 'zeebrugge', 'lisco', 'lakeerie', 'casablanca', 'irbelighthouse',
                 'ariaketower', 'kemigawa', 'uscseaprism', 'section7', 'southgreenbay', 'palgrunden', 'venezia', 'lucinda',
                 'bahiablanca', 'socheongcho', 'wavecissite', 'lakeokeechobee', 'gustavdalentower']
    else:
        workdir_ = '/sat_data/satellite/acix-iii'
        sites = ['Wendtorf', 'Varese', 'Geneve', 'Venice_Lagoon', 'Garda', 'Trasimeno']


    for site in sites:

        workdir = opj(workdir_, site)

        for l1c_path in glob.glob(opj(workdir, '*L1_STD_OFFL*.he5')):

            l1c = os.path.basename(l1c_path)

            l2c = l1c.replace('L1_STD_OFFL', 'L2C_STD')
            l2c_path = opj(workdir, l2c)

            # ---------------------------------------
            # construct L1C image plus angle rasters
            # ---------------------------------------
            try:
                # dc_l1c = driver.read_L1C_data(l1c_path, reflectance_unit=True, drop_vars=True)
                ds = h5py.File(l2c_path)
                lat = np.mean(ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Latitude"][:].T)
                lon = np.mean(ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Longitude"][:].T)
                date = ds.attrs["Product_StartTime"].decode('UTF-8')

            except:
                print('problem with file : ', l2c_path)

            # Geolocation


            info.append([l1c,lon,lat,date])

df = pd.DataFrame(info,columns=['file','lon','lat','date'])
df.to_csv('/sat_data/satellite/acix-iii/info_acix_prisma.csv',index=False)
df = pd.read_csv('/sat_data/satellite/acix-iii/info_acix_prisma.csv')

def get_cams(lon,lat,date,dir='./',
             type='cams-global-atmospheric-composition-forecasts',
             ):

    date_str = date.strftime('%Y-%m-%d')

    file = date_str + '-' + type + '.nc'


    # lazy loading
    cams = xr.open_dataset(opj(dir, file), decode_cf=True,
                           chunks={'time': 1, 'x': 500, 'y': 500})
    # slicing
    cams = cams.sel(time=date, method='nearest')
    return cams.sel(latitude=lat, longitude=lon,method='nearest')
odir='/DATA/projet/magellium/acix-iii/'
idir='/media/harmel/vol1/Dropbox/satellite/S2/cnes/CAMS'
for idx,_ in df.iterrows():
    date_str= _.date
    date_str = '2022-07-31T08:58:01.123000'
    ofile = _.file.replace('.he5','_cams.nc')
    date = dt.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
    rep = opj(idir,date.strftime('%Y/%m/%d'))
    print(rep)
    #cams = get_cams(_.lon,_.lat,date,dir='/media/harmel/vol1/Dropbox/satellite/S2/cnes/CAMS')
    #cams.to_netcdf(opj(odir,ofile))
