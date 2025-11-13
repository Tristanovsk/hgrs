import os
import numpy as np

import xarray as xr
import rioxarray as rxr

import xml.etree.ElementTree as ET
from hgrs import SolarIrradiance
import datetime
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
opj = os.path.join
from pyproj import CRS

def read_L1c_data (L1C_filepath: str,
                   metadata_filepath: str,
                   reflectance_unit=False,

                   drop_vars=False,
                   ):

    tree = ET.parse(metadata_filepath)
    root = tree.getroot()
    specific = root.find('specific')
    data = rxr.open_rasterio(L1C_filepath, chunks={'x':-1,'y':-1,'band':1}, mask_and_scale=True)
    date_str = specific.findtext('datatakeStart').strip().replace("Z", "")

    data = data.assign_coords({'wavelength': data.wavelength})
    data = data.set_index(band="wavelength")
    data = data.rename({'band': 'wl'})
    data.name = 'Ltoa'
    data = data.to_dataset()
    #data['time'] = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
    #data = data.set_coords('time')

    # convert from W.m-2.sr-1.nm-1 to mW.m-2.sr-1.nm-1
    data['Ltoa'] = 1e3 * data['Ltoa'].drop_attrs()
    data['Ltoa'].attrs['unit'] = 'mW.m-2.sr-1.nm-1'
    data['Ltoa'].attrs['description'] = 'top-of-atmosphere radiance'
    data['Ltoa'].attrs['name'] = 'radiance'

    x = data.x.values
    y = data.y.values

    x_coarse = [x[0], x[-1]]
    y_coarse = [y[0], y[-1]]
    XX, YY = np.meshgrid(x, y)
    points = np.stack([YY.ravel(), XX.ravel()], axis=-1)

    sza_values = np.zeros((2, 2))
    sza_values[0, 0] = 90 - float(specific.find('sunElevationAngle').findtext('upper_left'))
    sza_values[0, 1] = 90 - float(specific.find('sunElevationAngle').findtext('upper_right'))
    sza_values[1, 0] = 90 - float(specific.find('sunElevationAngle').findtext('lower_left'))
    sza_values[1, 1] = 90 - float(specific.find('sunElevationAngle').findtext('lower_right'))
    sza_interp = RegularGridInterpolator((y_coarse, x_coarse), sza_values, method='linear')
    sza = sza_interp(points).reshape(YY.shape)

    saa_values = np.zeros((2, 2))
    saa_values[0,0] = float(specific.find('sunAzimuthAngle').findtext('upper_left'))
    saa_values[0,1] = float(specific.find('sunAzimuthAngle').findtext('upper_right'))
    saa_values[1,0] = float(specific.find('sunAzimuthAngle').findtext('lower_left'))
    saa_values[1,1] = float(specific.find('sunAzimuthAngle').findtext('lower_right'))
    saa_interp = RegularGridInterpolator((y_coarse, x_coarse), saa_values, method='linear')
    saa = saa_interp(points).reshape(YY.shape)

    vza_values = np.zeros((2, 2))
    vza_values[0,0] = float(specific.find('viewingZenithAngle').findtext('upper_left'))
    vza_values[0,1] = float(specific.find('viewingZenithAngle').findtext('upper_right'))
    vza_values[1,0] = float(specific.find('viewingZenithAngle').findtext('lower_left'))
    vza_values[1,1] = float(specific.find('viewingZenithAngle').findtext('lower_right'))
    vza_interp = RegularGridInterpolator((y_coarse, x_coarse), vza_values, method='linear')
    vza = vza_interp(points).reshape(YY.shape)

    vaa_values = np.zeros((2, 2))
    vaa_values[0, 0] = float(specific.find('viewingAzimuthAngle').findtext('upper_left'))
    vaa_values[0, 1] = float(specific.find('viewingAzimuthAngle').findtext('upper_right'))
    vaa_values[1, 0] = float(specific.find('viewingAzimuthAngle').findtext('lower_left'))
    vaa_values[1, 1] = float(specific.find('viewingAzimuthAngle').findtext('lower_right'))
    vaa_interp = RegularGridInterpolator((y_coarse, x_coarse), vaa_values, method='linear')
    vaa = vaa_interp(points).reshape(YY.shape)

    raa = np.abs(saa-vaa)

    solar_irr = SolarIrradiance()
    F0 = solar_irr.tsis  # thuillier # gueymard # kurucz

    ## Compute irradiance for the Day Of the Year (date of acquisition)
    DOY = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday

    U = 1 - 0.01672 * np.cos(0.9856 * (DOY - 4))
    F0 = F0 * U
    F0_sensor = solar_irr.convolve(F0, data.fwhm, info={'description': 'Convolved solar irradiance from TSIS data',
                                                        'unit': 'mW/m2/nm'})

    data_final = xr.Dataset(data_vars=dict(Ltoa=(["wl", "y", "x"], data.Ltoa.values),
                                           F0=(['wl'], F0_sensor.values),
                                           fwhm=(['wl'], data.fwhm.values),
                                           sza = (["y", "x"], sza),
                                           saa = (["y", "x"], saa),
                                           vza = (["y", "x"], vza),
                                           vaa = (["y", "x"], vaa),
                                           raa = (['y', 'x'], raa),
                                           ),
                            coords=dict(
                                x=data.x.values,
                                y=data.y.values,
                                wl=data.wl.values,
                                time = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")),

                            attrs=dict(description="EnMAP L1C cube data"))
    ## Compute Top of Atmosphere Reflectance
    if reflectance_unit:
        cos_sza = xr.ufuncs.cos(np.radians(data_final.sza))
        data_final['Rtoa'] = np.pi * data_final.Ltoa / (data_final.F0.expand_dims({"y": data_final.y, "x": data_final.x}) * cos_sza.expand_dims({"wl": data_final.wl}))
        if drop_vars:
            data_final = data_final.drop_vars('Ltoa')

    # Attributes

    data_final.attrs["L1C_product_name"] = os.path.basename(L1C_filepath)
    data_final.attrs["acquisition_date"] = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
    data_final.sza.attrs['definition'] = " Sun Zenith Angle"
    data_final.saa.attrs['definition'] = " Sun Azimuth Angle"
    data_final.vza.attrs['definition'] = " Viewing Zenith Angle"
    data_final.vaa.attrs['definition'] = " Viewing Azimuth Angle"
    data_final.raa.attrs['definition'] = " Relative Azimuth Angle"

    data_final.sza.attrs['unit'] = " ° "
    data_final.saa.attrs['unit'] = " ° "
    data_final.vza.attrs['unit'] = " ° "
    data_final.vaa.attrs['unit'] = " ° "
    data_final.raa.attrs['unit'] = " ° "

    data_final.F0.attrs['unit'] = 'mW/m2/nm'
    data_final.Rtoa.attrs['definition'] = 'Reflectance at the Top of the Atmosphere'
    data_final.Rtoa.attrs['unit'] = '-'
    data_final.F0.attrs['definition'] = 'Solar irradiance corrected for Sun-Earth distance'
    data_final.Ltoa.attrs['unit'] = 'mW/m2/sr/nm'
    data_final.Ltoa.attrs['definition'] = 'Top-of-atmosphere radiance'
    data_final.fwhm.attrs['unit'] = 'nm'
    data_final.fwhm.attrs['definition'] = 'Full Width at Half Maximum'
    #data_final.attrs['crs'] = CRS.from_epsg(32630)




    return data_final

