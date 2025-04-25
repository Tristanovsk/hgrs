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
for case in [2] :#0,1]:
    if case == 0:
        workdir_ = '/data/acix-iii/Second_Batch/AERONET-OC'
        sites = ['galataplatform', 'sanmarcoplatform', 'zeebrugge', 'lisco', 'lakeerie', 'casablanca', 'irbelighthouse',
                 'ariaketower', 'kemigawa', 'uscseaprism', 'section7', 'southgreenbay', 'palgrunden', 'venezia', 'lucinda',
                 'bahiablanca', 'socheongcho', 'wavecissite', 'lakeokeechobee', 'gustavdalentower']
    elif case == 1:
        workdir_ = '/data/acix-iii/Second_Batch'
        sites = ['Wendtorf', 'Varese', 'Geneve', 'Venice_Lagoon', 'Garda', 'Trasimeno']
    else:
        workdir_ = '/data/acix-iii/Third_Batch'
        sites=['L1']

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
df.to_csv('/data/acix-iii/info_acix_prisma_v3.csv',index=False)

df = pd.read_csv('/data/acix-iii/info_acix_prisma_v3.csv')

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




def get_cams_v2(clon,
                clat,
                date,
                idir='./',
                type='cams_forecast',
                ):
    latmin, latmax = clat - 0.5, clat + 0.5
    lonmin, lonmax = clon - 0.5, clon + 0.5

    # get source CAMS file
    date_str = date.strftime('%Y-%m')
    file = type + '_' + date_str + '.nc'
    filepath = opj(idir, file)

    # lazy loading
    cams = xr.open_dataset(filepath, decode_cf=True,
                           chunks={'time': -1, 'x': 500, 'y': 500})
    cams = cams.sel(latitude=slice(latmax, latmin))
    # check if image is on Greenwich meridian and adapt longitude convention
    if cams.longitude.min() >= 0:
        if lonmin <= 0 and lonmax >= 0:

            cams = cams.assign_coords({"longitude": (((cams.longitude + 180) % 360) - 180)}).sortby('longitude')
        else:
            # set longitude between 0 and 360 deg
            lonmin, lonmax, = lonmin % 360, lonmax % 360

    # slicing
    cams = cams.sel(longitude=slice(lonmin, lonmax)).load()

    # rename "time" variable to avoid conflicts
    # cams = cams.rename({'time':'time_cams'})
    if cams.u10.shape[0] == 0 or cams.u10.shape[1] == 0:
        print('no cams data, enlarge subset')

    # get the nearest date
    return cams.sel(time=date, method='nearest')


odir='/DATA/projet/magellium/acix-iii/'
idir='/data/cams/world'
for idx,_ in df.iterrows():
    date_str= _.date

    ofile = _.file.replace('.he5','_cams.nc')
    date = dt.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
    print(ofile)
    cams = get_cams_v2(_.lon,_.lat,date,idir=idir).mean(['longitude','latitude'])
    cams.to_netcdf(opj(odir,ofile))
