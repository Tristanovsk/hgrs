#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:11:42 2023

@author: damali
"""
import os

import numpy as np
import xarray as xr
import rioxarray as rio

import h5py
import netCDF4 as nc
import scipy.interpolate
import scipy.integrate

import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cf

import datetime

opj = os.path.join

# ******************************************************************************************************
dir, filename = os.path.split(__file__)
solar_spectrum_file = opj(dir, '..', 'data', 'ref_atlas_thuillier3.nc')


def convolve_ISRF(init_wl: np.array, init_spectrum: np.array, new_wl: np.array, fwhm: np.array):
    # =============================================================================
    # Prepare stuff
    # =============================================================================
    # obj.I0 = convolve_ISRF(irr_wl, irr_dat, obj.wl, obj.isrf.fwhm, 'gaussmf')'
    if len(fwhm) == 1:
        fwhm = np.ones(new_wl.shape) * fwhm
    N_new_wl = len(new_wl)
    ksize = np.ones(new_wl.shape) * 200
    Bmin = new_wl - ksize * fwhm / 100
    Bmax = new_wl + ksize * fwhm / 100
    res = np.zeros(new_wl.shape)
    # =============================================================================
    # Perform convoluion for each new wavelength
    # =============================================================================
    for ii in range(N_new_wl):
        # Locate wavelenghts of the intitial spectra to perform the convolution
        ii_wl = (init_wl >= Bmin[ii]) & (init_wl <= Bmax[ii])
        x = init_wl[ii_wl]
        y = init_spectrum[ii_wl]
        # Calculate ISRF
        # ISRF = isrfCalculation(auxwl,C(i),FWHM(i),[],model);
        # function isrf = isrfCalculation(wl,C,FWHM,P,model)
        sig = fwhm[ii] / 2.3548
        ISRF = np.exp(-(x - new_wl[ii]) * (x - new_wl[ii]) / (2 * sig * sig))
        # Convolve data with ISRF
        res[ii] = scipy.integrate.simpson(y * ISRF, x) / scipy.integrate.simpson(ISRF, x)
        # plt.figure()
        # plt.plot(x,y,"b-.")
        # plt.plot(x, ISRF, "r--")
        # plt.plot(new_wl[ii], res[ii], "x", color="lime")
    return res


def load_thuillier_solar_spectrum(prisma_wl: np.array, prisma_fwhm: np.array):
    # =============================================================================
    # Load solar irradiance
    # =============================================================================
    ds = nc.Dataset(solar_spectrum_file)
    I0_mW = (ds["data"][:]).ravel()
    I0_wl = ds["wavelength"][:]
    # =============================================================================
    # Extrapolate to the SWIR
    # =============================================================================
    x_new = np.arange(I0_wl[-1], 2510.1, 0.1)
    extrap = scipy.interpolate.interp1d(I0_wl, I0_mW, kind="linear", fill_value="extrapolate")(x_new)
    # plt.plot(I0_wl, I0_mW, 'b--.')
    # plt.plot(x_new, extrap, 'r--x')
    I0_mW = np.append(I0_mW, extrap)
    I0_wl = np.append(I0_wl, x_new)
    # =============================================================================
    # Convolve with the PRISMA ISRF (gaussian)
    # =============================================================================
    I0_mW_conv = convolve_ISRF(I0_wl, I0_mW, prisma_wl, prisma_fwhm)
    # plt.plot(I0_wl, I0_mW, 'b--.')
    # plt.plot(x_new, extrap, 'r--.')
    # plt.plot(prisma_wl, I0_mW_conv, 'x', color="lime")
    return I0_mW_conv


def read_L1C_data(L1C_filepath: str,
                  reflectance_unit=False,
                  drop_vars=False):
    # =============================================================================
    # Load geolocation, solar irradiance and TOA radiance
    # =============================================================================
    ds = h5py.File(L1C_filepath)

    # coarse geometry
    sza = ds.attrs["Sun_zenith_angle"]

    # Geolocation
    lat = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR"][:]
    lon = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR"][:]
    xdim,ydim=lat.shape

    # Wavelength / fwhm
    wl = ds.attrs["List_Cw_Vnir"][5:]
    wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
    fwhm = ds.attrs["List_Fwhm_Vnir"][5:]
    fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
    sort_index = np.argsort(wl)
    wl = wl[sort_index]
    fwhm = fwhm[sort_index]

    # Thuillier solar irradiance convolved to the PRISMA ISRF and scaled
    # by the day of the year
    I0 = load_thuillier_solar_spectrum(wl, fwhm) * 1e3
    DOY = datetime.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),
                                     "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
    U = 1 - 0.01672 * np.cos(0.9856 * (DOY - 4))
    I0 = I0 * U
    # DN to TOA radiance
    gain = {"vnir": ds.attrs["ScaleFactor_Vnir"],
            "swir": ds.attrs["ScaleFactor_Swir"]}

    # -------------------------------------------------------------------------------
    VNIR = np.moveaxis(ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"][:, 5:, :] / gain["vnir"],
                       [0, 1, 2],
                       [1, 2, 0])
    SWIR = np.moveaxis(ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][:, :-2, :] / gain["swir"],
                       [0, 1, 2],
                       [1, 2, 0])
    Ltoa = np.dstack((VNIR, SWIR))
    del VNIR, SWIR

    Ltoa = Ltoa[:, :, sort_index]

    data = xr.Dataset(data_vars=dict(Ltoa=(["y", "x", "wl"], Ltoa), F0=(['wl'], I0)),
                      coords=dict(
                          x=range(xdim),
                          y=range(ydim),
                          lon=(["x", "y"], lon),
                          lat=(["x", "y"], lat),
                          wl=wl),
                      attrs=dict(description="PRISMA L1C cube data"))
    # TODO check errors due to bulk SZA value instead of per pixel values
    if reflectance_unit:
        data['Rtoa'] = np.pi * data.Ltoa / (data.F0 * np.cos(np.radians(sza)))
        if drop_vars:
            data = data.drop_vars('Ltoa')

    # =============================================================================
    # Load other metadata
    # =============================================================================
    data.attrs["L1C_product_name"] = os.path.basename(L1C_filepath)
    data.attrs["acquisition_date"] = ds.attrs["Product_StartTime"].decode('UTF-8')
    data.attrs["sza"] = ds.attrs["Sun_zenith_angle"]
    data.attrs["saa"] = ds.attrs["Sun_azimuth_angle"]
    data.F0.attrs['unit'] = 'mW/m2/nm'
    data.F0.attrs['definition'] = 'Solar irradiance corrected for Sun-Earth distance'

    # =============================================================================
    # Load masks
    # =============================================================================
    data = data.assign(cloud_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/Cloud_Mask"][:]))
    data = data.assign(sunglint_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SunGlint_Mask"][:]))
    data = data.assign(landcover_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/LandCover_Mask"][:]))
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
    wl = ds.attrs["List_Cw_Vnir"][3:]
    wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
    fwhm = ds.attrs["List_Fwhm_Vnir"][3:]
    fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
    sort_index = np.argsort(wl)
    wl = wl[sort_index]
    fwhm = fwhm[sort_index]
    # Thuillier solar irradiance convolved to the PRISMA ISRF and scaled
    # by the day of the year
    I0 = load_thuillier_solar_spectrum(wl, fwhm)
    DOY = datetime.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),
                                     "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
    U = 1 - 0.01672 * np.cos(0.9856 * (DOY - 4))
    I0 = I0 * U
    # DN to TOA radiance
    gain = {"vnir_min": ds.attrs["L2ScaleVnirMin"],
            "vnir_max": ds.attrs["L2ScaleVnirMax"],
            "swir_min": ds.attrs["L2ScaleSwirMin"],
            "swir_max": ds.attrs["L2ScaleSwirMax"]}

    # -------------------------------------------------------------------------------
    VNIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/VNIR_Cube"][:, 3:, :]
    VNIR = gain["vnir_min"] + VNIR * (gain["vnir_max"] - gain["vnir_min"]) / 65535
    VNIR = np.moveaxis(VNIR, [0, 1, 2], [1, 2, 0])
    SWIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube"][:, :-2, :]
    SWIR = gain["swir_min"] + SWIR * (gain["swir_max"] - gain["swir_min"]) / 65535
    SWIR = np.moveaxis(SWIR, [0, 1, 2], [1, 2, 0])
    # print(f"VNIR shape = {VNIR.shape}")
    # print(f"SWIR shape = {SWIR.shape}")
    rho = np.dstack((VNIR, SWIR))
    rho = rho[:, :, sort_index]
    del VNIR, SWIR

    # # -------------------------------------------------------------------------------
    data = xr.Dataset(data_vars=dict(rho=(["y", "x", "wl"], rho)),
                      coords=dict(
                          lon=(["x", "y"], lon),
                          lat=(["x", "y"], lat),
                          wl=wl),
                      attrs=dict(description="RISMA L2C cube data"))

    # =============================================================================
    # Load other metadata
    # =============================================================================
    data.attrs["L2C_product_name"] = os.path.basename(L2C_filepath)
    data.attrs["acquisition_date"] = ds.attrs["Product_StartTime"].decode('UTF-8')

    # =============================================================================
    # Load geometries 
    # =============================================================================
    data = data.assign(vza=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Observing_Angle"][:]))
    data = data.assign(raa=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Rel_Azimuth_Angle"][:]))
    data = data.assign(sza=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Solar_Zenith_Angle"][:]))
    # =============================================================================
    # Read atmospheric data
    # =============================================================================
    hdf_variables = ["AOT", "AEX", "WVM", "COT"]
    ds_variables = ["aot", "aex", "wvm", "cot"]
    dims = {"AOT": ["y2", "x2"],
            "AEX": ["y2", "x2"],
            "WVM": ["y", "x"],
            "COT": ["y", "x"]}
    for ii, var in enumerate(hdf_variables):
        gain_min = ds.attrs[f"L2Scale{var}Min"]
        gain_max = ds.attrs[f"L2Scale{var}Max"]
        matrix = ds[f"/HDFEOS/SWATHS/PRS_L2C_{var}/Data Fields/{var}_Map"][:]
        var_dims = dims[var]
        data = eval(f'data.assign({ds_variables[ii]}=({var_dims},gain_min + matrix*(gain_max-gain_min)/65535))')
    return data

