
import os

import numpy as np
import pandas as pd
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

from hgrs import solar_irradiance




def read_L1C_data(L1C_filepath: str,
                  reflectance_unit=False,
                  drop_vars=False):
    '''
    Load PRISMA L1C data into xarray rasters
    :param L1C_filepath: absolute path to the .h5 prisma file
    :param reflectance_unit: to convert from TOA radiance to TOA reflectance
    :param drop_vars: if True remove the radiance raster to keep reflectance only
    :return:
    '''
    # =============================================================================
    # Load geolocation, solar irradiance and TOA radiance
    # =============================================================================
    ds = h5py.File(L1C_filepath)

    # coarse geometry
    sza = ds.attrs["Sun_zenith_angle"]

    # Geolocation
    lat = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR"][:].T
    lon = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR"][:].T
    xdim, ydim = lat.shape

    # Wavelength / fwhm
    wl = ds.attrs["List_Cw_Vnir"][5:]
    wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
    fwhm = ds.attrs["List_Fwhm_Vnir"][5:]
    fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
    sort_index = np.argsort(wl)
    wl = wl[sort_index]
    fwhm = fwhm[sort_index]
    fwhm = xr.DataArray(data=fwhm, name='fwhm',
                        coords=dict(wl=wl),
                        attrs=dict(description="PRISMA relative spectral response parameter"))

    # solar irradiance convolution to the PRISMA spectral response function and scaled
    # by the day of the year
    solar_irr = solar_irradiance()
    F0 = solar_irr.tsis #huillier # gueymard # kurucz
    DOY = datetime.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),
                                     "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
    U = 1 - 0.01672 * np.cos(0.9856 * (DOY - 4))
    F0 = F0 * U
    F0_sensor = solar_irr.convolve(F0, fwhm, info={'description': 'Convolved solar irradiance from TSIS data',
                                                   'unit': 'mW/m2/nm'})
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

    data = xr.Dataset(data_vars=dict(Ltoa=(["y", "x", "wl"], Ltoa),
                                     F0=(['wl'], F0_sensor.values),
                                     fwhm=(['wl'], fwhm.values),
                                     lon=(["y", "x"], lon),
                                     lat=(["y", "x"], lat)),
                      coords=dict(
                          x=np.arange(xdim)[::-1],
                          y=np.arange(ydim)[::-1],

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
    data = data.assign(cloud_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/Cloud_Mask"][:].T))
    data = data.assign(sunglint_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SunGlint_Mask"][:].T))
    data = data.assign(landcover_mask=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/LandCover_Mask"][:].T))
    return data


def read_L2C_data(L2C_filepath: str):
    '''
    Load PRISMA L2C data (including observation angles) into xarray rasters
    :param L2C_filepath: absolute path to the .h5 prisma file
    :return:
    '''

    # =============================================================================
    # Load geolocation, solar irradiance and TOA radiance
    # =============================================================================
    ds = h5py.File(L2C_filepath)
    # Geolocation
    lat = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Latitude"][:].T
    lon = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Longitude"][:].T
    xdim, ydim = lat.shape

    # Wavelength / fwhm
    wl = ds.attrs["List_Cw_Vnir"][3:]
    wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
    fwhm = ds.attrs["List_Fwhm_Vnir"][3:]
    fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
    sort_index = np.argsort(wl)
    wl = wl[sort_index]
    fwhm = fwhm[sort_index]

    # # Thuillier solar irradiance convolved to the PRISMA ISRF and scaled
    # # by the day of the year
    # I0 = load_thuillier_solar_spectrum(wl, fwhm)
    # DOY = datetime.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),
    #                                  "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
    # U = 1 - 0.01672 * np.cos(0.9856 * (DOY - 4))
    # I0 = I0 * U

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
    data = xr.Dataset(data_vars=dict(rho=(["y", "x", "wl"], rho), lon=(["y", "x"], lon),
                                     lat=(["y", "x"], lat)),
                      coords=dict(
                          x=np.arange(xdim)[::-1],
                          y=np.arange(ydim)[::-1],

                          wl=wl),
                      attrs=dict(description="PRISMA L2C cube data"))

    # =============================================================================
    # Load other metadata
    # =============================================================================
    data.attrs["L2C_product_name"] = os.path.basename(L2C_filepath)
    data.attrs["acquisition_date"] = ds.attrs["Product_StartTime"].decode('UTF-8')

    # =============================================================================
    # Load geometries 
    # =============================================================================
    data = data.assign(vza=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Observing_Angle"][:].T))
    data = data.assign(raa=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Rel_Azimuth_Angle"][:].T))
    data = data.assign(sza=(["y", "x"], ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Solar_Zenith_Angle"][:].T))

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
        matrix = ds[f"/HDFEOS/SWATHS/PRS_L2C_{var}/Data Fields/{var}_Map"][:].T
        var_dims = dims[var]
        data = eval(f'data.assign({ds_variables[ii]}=({var_dims},gain_min + matrix*(gain_max-gain_min)/65535))')
    x2dim, y2dim = data['aot'].shape
    data = data.assign_coords({'x2': np.arange(x2dim)[::-1], 'y2': np.arange(y2dim)[::-1]})

    return data
