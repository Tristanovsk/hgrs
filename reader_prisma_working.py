#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:11:42 2023

@author: damali
"""

import h5py
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cf
import numpy as np
import netCDF4 as nc
import scipy.interpolate
import scipy.integrate
import datetime
import rioxarray as rio
import os


# ******************************************************************************************************


def convolve_ISRF(init_wvl: np.array, init_spectrum: np.array, new_wvl: np.array, fwhm: np.array):
    # =============================================================================
    # Prepare stuff
    # =============================================================================
    # obj.I0 = convolve_ISRF(irr_wvl, irr_dat, obj.wvl, obj.isrf.fwhm, 'gaussmf')'
    if len(fwhm)==1:
        fwhm = np.ones(new_wvl.shape)*fwhm
    N_new_wvl = len(new_wvl)
    ksize = np.ones(new_wvl.shape)*200
    Bmin = new_wvl - ksize*fwhm/100
    Bmax = new_wvl + ksize*fwhm/100
    res = np.zeros(new_wvl.shape)
    # =============================================================================
    # Perform convoluion for each new wavelength
    # =============================================================================
    for ii in range(N_new_wvl):
        # Locate wavelenghts of the intitial spectra to perform the convolution
        ii_wvl = (init_wvl>=Bmin[ii]) & (init_wvl<=Bmax[ii])
        x      = init_wvl[ii_wvl]
        y      = init_spectrum[ii_wvl]
        # Calculate ISRF
        # ISRF = isrfCalculation(auxwvl,C(i),FWHM(i),[],model);
        # function isrf = isrfCalculation(wvl,C,FWHM,P,model)
        sig  = fwhm[ii]/2.3548
        ISRF = np.exp(-(x-new_wvl[ii])*(x-new_wvl[ii])/(2*sig*sig))
        # Convolve data with ISRF
        res[ii] = scipy.integrate.simpson(y*ISRF,x)/scipy.integrate.simpson(ISRF,x)
        # plt.figure()
        # plt.plot(x,y,"b-.")
        # plt.plot(x, ISRF, "r--")
        # plt.plot(new_wvl[ii], res[ii], "x", color="lime")
    return res


def load_thuillier_solar_spectrum(prisma_wvl: np.array, prisma_fwhm: np.array):
    # =============================================================================
    # Load solar irradiance
    # =============================================================================
    ds = nc.Dataset("/home/damali/Work/CMIX/ref_atlas_thuillier3.nc")
    I0_mW = (ds["data"][:]).ravel()
    I0_wvl = ds["wavelength"][:]
    # =============================================================================
    # Extrapolate to the SWIR
    # =============================================================================
    x_new = np.arange(I0_wvl[-1],2510.1,0.1)
    extrap = scipy.interpolate.interp1d(I0_wvl,I0_mW,kind="linear",fill_value="extrapolate")(x_new)
    # plt.plot(I0_wvl, I0_mW, 'b--.')
    # plt.plot(x_new, extrap, 'r--x')
    I0_mW = np.append(I0_mW, extrap)
    I0_wvl = np.append(I0_wvl, x_new)
    # =============================================================================
    # Convolve with the PRISMA ISRF (gaussian)
    # =============================================================================
    I0_mW_conv = convolve_ISRF(I0_wvl, I0_mW, prisma_wvl, prisma_fwhm)
    # plt.plot(I0_wvl, I0_mW, 'b--.')
    # plt.plot(x_new, extrap, 'r--.')
    # plt.plot(prisma_wvl, I0_mW_conv, 'x', color="lime")
    return I0_mW_conv


def read_L1C_data(L1C_filepath: str):
    # =============================================================================
    # Load geolocation, solar irradiance and TOA radiance
    # =============================================================================
    ds = h5py.File(L1C_filepath)
    # Geolocation
    lat = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR"][:]
    lon = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR"][:]
    # Wavelength / fwhm
    wvl = ds.attrs["List_Cw_Vnir"][5:]
    wvl = np.append(wvl, ds.attrs["List_Cw_Swir"][:-2])
    fwhm = ds.attrs["List_Fwhm_Vnir"][5:]
    fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
    sort_index = np.argsort(wvl)
    wvl = wvl[sort_index]
    fwhm = fwhm[sort_index]
    # Thuillier solar irradiance convolved to the PRISMA ISRF and scaled
    # by the day of the year
    I0 = load_thuillier_solar_spectrum(wvl, fwhm)
    DOY = datetime.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),"%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
    U = 1 - 0.01672*np.cos(0.9856*(DOY-4))
    I0 = I0*U
    # DN to TOA radiance
    gain = {"vnir":ds.attrs["ScaleFactor_Vnir"],
            "swir":ds.attrs["ScaleFactor_Swir"]}
    # -------------------------------------------------------------------------------
    # fig = plt.figure()
    # ax = fig.add_axes(plt.axes(projection=crs.PlateCarree()))
    # ax.plot([lon[0,0],lon[0,-1],lon[-1,-1],lon[-1,0],lon[0,0]],
    #         [lat[0,0],lat[0,-1],lat[-1,-1],lat[-1,0],lat[0,0]],
    #         'r-o')
    # ax.add_feature(cf.COASTLINE, linewidth=0.3)
    # ax.add_feature(cf.BORDERS, linewidth=0.3, linestyle=":")
    # ax.add_feature(cf.LAND)
    # ax.add_feature(cf.OCEAN)
    # ax.add_feature(cf.LAKES)
    # ax.add_feature(cf.RIVERS)
    # ax.set_extent([np.min(lon)-10,np.max(lon)+10,np.min(lat)-10,np.max(lat)+10])
    # -------------------------------------------------------------------------------
    VNIR = np.moveaxis(ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"][:,5:,:] / gain["vnir"],
                       [0,1,2],
                       [1,2,0])
    SWIR = np.moveaxis(ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][:,:-2,:] / gain["swir"],
                       [0,1,2],
                       [1,2,0])
    Ltoa = np.dstack((VNIR, SWIR))
    Ltoa = Ltoa[:,:,sort_index]
    del VNIR, SWIR
    # -------------------------------------------------------------------------------
    # fig = plt.figure()
    # ax = fig.add_axes(plt.axes(projection=crs.PlateCarree()))
    # ax.scatter(lon,
    #            lat,
    #            c=VNIR[:,15,:],
    #            cmap="turbo")
    # ax.add_feature(cf.COASTLINE, linewidth=0.3)
    # ax.add_feature(cf.BORDERS, linewidth=0.3, linestyle=":")
    # ax.add_feature(cf.LAND)
    # ax.add_feature(cf.OCEAN)
    # ax.add_feature(cf.LAKES)
    # ax.add_feature(cf.RIVERS)
    # ax.set_extent([np.min(lon)-10,np.max(lon)+10,np.min(lat)-10,np.max(lat)+10])
    # -------------------------------------------------------------------------------
    data = xr.Dataset(data_vars=dict(Ltoa=(["y1","x1","wvl"], Ltoa)),
                      coords=dict(
                          lon=(["x1","y1"],lon),
                          lat=(["x1","y1"],lat),
                          wvl=wvl),
                      attrs=dict(description="PRISMA L1C cube data"))
    # data.Ltoa[:,:,15].plot(x="lon",y="lat")
    # =============================================================================
    # Load other metadata
    # =============================================================================
    data.attrs["L1C_product_name"] = os.path.basename(L1C_filepath)
    data.attrs["acquisition_date"] = ds.attrs["Product_StartTime"].decode('UTF-8')
    data.attrs["sza"] = ds.attrs["Sun_zenith_angle"]
    data.attrs["saa"] = ds.attrs["Sun_azimuth_angle"]
    # =============================================================================
    # Load masks
    # =============================================================================
    data = data.assign(cloud_mask=(["y1","x1"],ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/Cloud_Mask"][:]))
    data = data.assign(sunglint_mask=(["y1","x1"],ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SunGlint_Mask"][:]))
    data = data.assign(landcover_mask=(["y1","x1"],ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/LandCover_Mask"][:]))
    return data


def read_L2C_data(L2C_filepath: str):
    # L2C_filepath = "/home/damali/Work/CMIX/PRS_L2C_STD_20220412100139_20220412100143_0001_ROM.he5"
    # =============================================================================
    # Load geolocation, solar irradiance and TOA radiance
    # =============================================================================
    ds = h5py.File(L2C_filepath)
    # Geolocation
    lat = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Latitude"][:]
    lon = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Longitude"][:]
    # Wavelength / fwhm
    wvl = ds.attrs["List_Cw_Vnir"][3:]
    wvl = np.append(wvl, ds.attrs["List_Cw_Swir"][:-2])
    fwhm = ds.attrs["List_Fwhm_Vnir"][3:]
    fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
    sort_index = np.argsort(wvl)
    wvl = wvl[sort_index]
    fwhm = fwhm[sort_index]
    # Thuillier solar irradiance convolved to the PRISMA ISRF and scaled
    # by the day of the year
    I0 = load_thuillier_solar_spectrum(wvl, fwhm)
    DOY = datetime.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),"%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
    U = 1 - 0.01672*np.cos(0.9856*(DOY-4))
    I0 = I0*U
    # DN to TOA radiance
    gain = {"vnir_min":ds.attrs["L2ScaleVnirMin"],
            "vnir_max":ds.attrs["L2ScaleVnirMax"],
            "swir_min":ds.attrs["L2ScaleSwirMin"],
            "swir_max":ds.attrs["L2ScaleSwirMax"]}
    # -------------------------------------------------------------------------------
    # fig = plt.figure()
    # ax = fig.add_axes(plt.axes(projection=crs.PlateCarree()))
    # ax.plot([lon[0,0],lon[0,-1],lon[-1,-1],lon[-1,0],lon[0,0]],
    #         [lat[0,0],lat[0,-1],lat[-1,-1],lat[-1,0],lat[0,0]],
    #         'r-o')
    # ax.add_feature(cf.COASTLINE, linewidth=0.3)
    # ax.add_feature(cf.BORDERS, linewidth=0.3, linestyle=":")
    # ax.add_feature(cf.LAND)
    # ax.add_feature(cf.OCEAN)
    # ax.add_feature(cf.LAKES)
    # ax.add_feature(cf.RIVERS)
    # ax.set_extent([np.min(lon)-10,np.max(lon)+10,np.min(lat)-10,np.max(lat)+10])
    # -------------------------------------------------------------------------------
    VNIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/VNIR_Cube"][:,3:,:]
    VNIR = gain["vnir_min"] + VNIR*(gain["vnir_max"]-gain["vnir_min"])/65535
    VNIR = np.moveaxis(VNIR, [0,1,2], [1,2,0])
    SWIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube"][:,:-2,:]
    SWIR = gain["swir_min"] + SWIR*(gain["swir_max"]-gain["swir_min"])/65535
    SWIR = np.moveaxis(SWIR, [0,1,2], [1,2,0])
    # print(f"VNIR shape = {VNIR.shape}")
    # print(f"SWIR shape = {SWIR.shape}")
    rho = np.dstack((VNIR, SWIR))
    rho = rho[:,:,sort_index]
    del VNIR, SWIR
    # # -------------------------------------------------------------------------------
    # # fig = plt.figure()
    # # ax = fig.add_axes(plt.axes(projection=crs.PlateCarree()))
    # # ax.scatter(lon,
    # #            lat,
    # #            c=VNIR[:,15,:],
    # #            cmap="turbo")
    # # ax.add_feature(cf.COASTLINE, linewidth=0.3)
    # # ax.add_feature(cf.BORDERS, linewidth=0.3, linestyle=":")
    # # ax.add_feature(cf.LAND)
    # # ax.add_feature(cf.OCEAN)
    # # ax.add_feature(cf.LAKES)
    # # ax.add_feature(cf.RIVERS)
    # # ax.set_extent([np.min(lon)-10,np.max(lon)+10,np.min(lat)-10,np.max(lat)+10])
    # # -------------------------------------------------------------------------------
    data = xr.Dataset(data_vars=dict(rho=(["y1","x1","wvl"], rho)),
                      coords=dict(
                          lon=(["x1","y1"],lon),
                          lat=(["x1","y1"],lat),
                          wvl=wvl),
                      attrs=dict(description="RISMA L2C cube data"))
    # data.Ltoa[:,:,15].plot(x="lon",y="lat")
    # =============================================================================
    # Load other metadata
    # =============================================================================
    data.attrs["L2C_product_name"] = os.path.basename(L2C_filepath)
    data.attrs["acquisition_date"] = ds.attrs["Product_StartTime"].decode('UTF-8')
    # =============================================================================
    # Load geometries 
    # =============================================================================
    data = data.assign(vza=(["y1","x1"],ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Observing_Angle"][:]))
    data = data.assign(vaa=(["y1","x1"],ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Rel_Azimuth_Angle"][:]))
    data = data.assign(sza=(["y1","x1"],ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Solar_Zenith_Angle"][:]))
    # =============================================================================
    # Read atmospheric data
    # =============================================================================
    hdf_variables = ["AOT","AEX","WVM","COT"]
    ds_variables  = ["aot","aex","wvm","cot"]
    dims = {"AOT":["y2","x2"],
            "AEX":["y2","x2"],
            "WVM":["y1","x1"],
            "COT":["y1","x1"]}
    for ii,var in enumerate(hdf_variables):
        gain_min = ds.attrs[f"L2Scale{var}Min"]
        gain_max = ds.attrs[f"L2Scale{var}Max"]
        matrix   = ds[f"/HDFEOS/SWATHS/PRS_L2C_{var}/Data Fields/{var}_Map"][:]
        var_dims = dims[var]
        data = eval(f'data.assign({ds_variables[ii]}=({var_dims},gain_min + matrix*(gain_max-gain_min)/65535))')
    return data


# ******************************************************************************************************

if __name__=="__main__":
    L1Cpath = "/home/damali/Work/CMIX/PRS_L1_STD_OFFL_20220701111705_20220701111710_0001.he5"
    L2Cpath = "/home/damali/Work/CMIX/PRS_L2C_STD_20220412100139_20220412100143_0001_ROM.he5"
    
    ds_L1C = read_L1C_data(L1Cpath)
    ds_L2C = read_L2C_data(L2Cpath)
    
    # =============================================================================
    # Crop dataset to a bounding box
    # =============================================================================
    lat_min, lat_max, lon_min, lon_max = 47.20, 47.30, -2.4, -2.2
    ds_L1C_cropped = ds_L1C.where((ds_L1C.lat<lat_max) & (ds_L1C.lat>lat_min) & (ds_L1C.lon>lon_min) & (ds_L1C.lon<lon_max))
    
    # =============================================================================
    # Plot wavelength closest to the given one
    # =============================================================================
    wvl0 = 415 # nm
    ds_L1C.sel(wvl=wvl0, method="nearest").Ltoa.plot(x="lon",y="lat")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    