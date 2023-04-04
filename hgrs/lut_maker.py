


import os, copy
import glob

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
plt.ioff()
import colorcet as cc

import numpy as np
import pandas as pd
import xarray as xr
from numba import jit

import hgrs.driver as driver
import hgrs

opj = os.path.join

#---------------------------------------
# open a L1C PRISMA image to get spectral response function of thesensor (e.g., fwhm)
#---------------------------------------
workdir_ = '/sat_data/satellite/acix-iii/'
site='Geneve'
workdir=opj(workdir_,site)
l1c='PRS_L1_STD_OFFL_20210906103712_20210906103717_0001.he5'
l1c_path=opj(workdir,l1c)
dc_l1c = driver.read_L1C_data(l1c_path,reflectance_unit=True,drop_vars=True)
prod = hgrs.algo(dc_l1c)
prod.load_metadata()

#---------------------------------------
# generate LUT
#---------------------------------------
gas_lut=prod.gas_lut
wl_ref = prod.gas_lut.wl.values
prisma_rsr = prod.raster.fwhm #.to_dataframe()
fwhm = prisma_rsr.values
wl_sat = prisma_rsr.wl.values

# Water vapor transmittance
air_masses = np.array([*np.linspace(2,6,41),6.5,7.,7.5,8.,9,10,11,12,13,14,15,20,30])
tcwvs = np.array([0 ,1 ,2,5 ,7.5,10 ,12.5,15 ,20 ,25 ,30 ,35,40,45,50 ,60])
ot_h2o = gas_lut.h2o.values
Nair_mass = air_masses.shape[0]
Ntcwv = tcwvs.shape[0]
Nwl = wl_sat.shape[0]

Twv =np.zeros((Nair_mass,Ntcwv,Nwl))

@jit(nopython=True)
def wv_transmittance(Twv,air_masses,tcwvs):

    def Gamma2sigma(Gamma):
        '''Function to convert FWHM (Gamma) to standard deviation (sigma)'''
        return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)

    def gaussian(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


    for i_air_mass in range(Nair_mass):
        print(i_air_mass)
        for i_tcwv in range(Ntcwv):
            ot_wv = ot_h2o *tcwvs[i_tcwv]
            Ttot = np.exp(-air_masses[i_air_mass] * ot_wv)
            for i_fwhm in range(Nwl):
                sig = Gamma2sigma(fwhm[i_fwhm])
                rsr = gaussian(wl_ref, wl_sat[i_fwhm], sig)
                Twv[i_air_mass,i_tcwv,i_fwhm] = np.trapz(Ttot * rsr,wl_ref) / np.trapz(rsr, wl_ref)

    return Twv

Twv = wv_transmittance(Twv,air_masses,tcwvs)

xTwv = xr.Dataset(dict(Twv=(["air_mass", "tcwv","wl"], Twv)),
           coords=dict(air_mass=air_masses,
                       tcwv=tcwvs,
                       wl=wl_sat ),
           attrs=dict(
              description="LUT for water vapor transmittances of PRISMA sensor",
              units="")
          )

xTwv.to_netcdf('data/lut/water_vapor_transmittance.nc')