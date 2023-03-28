#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, copy
import glob

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr
import rioxarray as rxr

import netCDF4
import h5py
from osgeo import gdal

#%matplotlib widget
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc

import hgrs.driver as driver
import hgrs

opj = os.path.join
hgrs.__version__


# In[2]:


auxdir = hgrs.__path__[0]
auxdir


# In[3]:


workdir = '/sat_data/satellite/acix-iii/AERONET-OC/bahiablanca'
l1c = 'PRS_L1_STD_OFFL_20210814141750_20210814141755_0001.he5'
#workdir = '/sat_data/satellite/acix-iii/Garda'
#l1c = 'PRS_L1_STD_OFFL_20210721102700_20210721102705_0001.he5'
l2c = l1c.replace('L1_STD_OFFL','L2C_STD')


# In[4]:


# parameters
wl_water_vapor=slice(800,1300)
wl_glint=slice(2100,2200)
wl_atmo = slice(950,2250)
wl_to_remove = [(935,967),(1105,1170),(1320,1490),(1778,2033)]
xcoarsen=20
ycoarsen=20
block_size = 2
abs_gas_file = '/DATA/git/vrtc/libradtran_tbx/output/lut_abs_opt_thickness_normalized.nc' 
lut_file = '/media/harmel/vol1/work/git/vrtc/RTxploitation/study_cases/aerosol/lut/opac_osoaa_lut_v2.nc'
pressure_rot_ref=1013.25

# TODO get them from CAMS
to3c=6.5e-3
tno2c=3e-6
tch4c= 1e-2
psl=1013
coef_abs_scat=0.3


# In[5]:


l1c_path = opj(workdir,l1c)
l2c_path = opj(workdir,l2c)

dc_l1c = driver.read_L1C_data(l1c_path,reflectance_unit=True,drop_vars=True)
dc_l2c = driver.read_L2C_data(l2c_path)


# ## Load metadata

# In[6]:


gas_lut =xr.open_dataset(abs_gas_file)
aero_lut = xr.open_dataset(lut_file)


# ## Add angle data into L1C object

# In[7]:


for param in ['sza','vza','raa']:
    dc_l1c[param]=dc_l2c[param]


# In[8]:


dc_l1c


# In[9]:


coarsening=1
gamma=0.5
brightness_factor = 1
plt.figure(figsize=(5,5))
fig = (dc_l1c.Rtoa[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])**gamma*brightness_factor).plot.imshow(rgb='wl',robust=True)#, subplot_kws=dict(projection= l1c.proj))
fig.axes.set(xticks=[], yticks=[])
fig.axes.set_ylabel('')
fig.axes.set_xlabel('')
fig


# ## First, proceed with the water pixel masking.

# ### Check prisma l1c masks

# In[10]:


fig,axs = plt.subplots(1,3,figsize=(16, 4))
fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.99,
                    hspace=0.15, wspace=0.15)
shrink = 0.8
axs=axs.ravel()    

for i,param in enumerate(['cloud_mask','sunglint_mask','landcover_mask']):

    dc_l1c[param][::coarsening, ::coarsening].plot.imshow(cmap=plt.cm.Spectral_r, vmax=60,robust=True,
                               cbar_kwargs={'shrink': shrink},ax=axs[i]) # extent=extent_val, transform=proj, 
    axs[i].set(xticks=[], yticks=[])
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')    
    axs[i].set_title(param)                     


# In[11]:


# Compute NDWI
img=dc_l1c.Rtoa
green = img.sel(wl=slice(540,570)).mean(dim='wl')
nir = img.sel(wl=slice(850,890)).mean(dim='wl')
ndwi = (green - nir) / (green + nir)


# In[12]:


coarsening=1

# binary cmap
bcmap = mpl.colors.ListedColormap(['khaki', 'lightblue'])

def water_mask(ndwi, threshold=0):
    water = xr.where(ndwi > threshold, 1, 0)
    return water.where(~np.isnan(ndwi))

def plot_water_mask(ndwi,ax,threshold=0):
    water = water_mask(ndwi, threshold)
    #ax.set_extent(extent_val, proj)
    water.plot.imshow(cmap=bcmap,ax=ax,
                      cbar_kwargs={'ticks': [0, 1], 'shrink': shrink})#extent=extent_val, transform=proj,
    ax.set_title(str(threshold)+' < NDWI')
    
fig,axs = plt.subplots(1,4,figsize=(22, 4))
fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.99,
                    hspace=0.1, wspace=0.15)
shrink = 0.8
axs=axs.ravel()    


fig = ndwi[::coarsening, ::coarsening].plot.imshow(cmap=plt.cm.BrBG, robust=True,
                                   cbar_kwargs={'shrink': shrink},ax=axs[0])# extent=extent_val, transform=proj, 

axs[0].set_title('PRISMA, NDWI')

for i,threshold in enumerate([-0.1,0.,0.1]):    
    plot_water_mask(ndwi[::coarsening, ::coarsening],axs[i+1],threshold=threshold)
for i in range(4):    
    axs[i].set(xticks=[], yticks=[])
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')    
plt.show()



# ## Mask pixels based on NDWI threshold

# In[13]:


ndwi_threshold=0.1

masked = dc_l1c.copy()
param_tomask = ['Rtoa']
for param in param_tomask:
    masked[param] = masked[param].where(ndwi > ndwi_threshold)


# In[14]:


masked
coarsening=1
gamma=0.2
brightness_factor = 1
plt.figure(figsize=(5,5))
fig = (masked.Rtoa[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])**gamma*brightness_factor).plot.imshow(rgb='wl',robust=True)#, subplot_kws=dict(projection= l1c.proj))
fig.axes.set(xticks=[], yticks=[])
fig.axes.set_ylabel('')
fig.axes.set_xlabel('')
fig
plt.show()


# In[15]:


# check maximum size of angle combinations to reconstruct from LUT
res=1
#len(np.unique(dc_l2c.sza.round(res)))*len(np.unique(dc_l2c.vza.round(res)))*len(np.unique(dc_l2c.raa.round(res)))


# In[16]:


fig,axs= plt.subplots(1,4,figsize=(22,4))
masked.sza.round(res).plot.imshow(ax=axs[0],robust=True)
masked.vza.round(res).plot.imshow(ax=axs[1],robust=True)
masked.raa.round(res).plot.imshow(ax=axs[2],robust=True)
air_mass = 1./np.cos(np.radians(masked.sza))+1./np.cos(np.radians(masked.vza))

#air_mass.plot.imshow(ax=axs[0],robust=True)
air_mass.round(3).plot.imshow(ax=axs[3],robust=True)
for i in range(4):    
    axs[i].set(xticks=[], yticks=[])
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')    


# In[17]:


M = np.unique(air_mass.round(3))


# ## Load sensor spectral responses

# In[18]:


prisma_rsr = dc_l1c.fwhm.to_dataframe()


# In[19]:


def Gamma2sigma(Gamma):
    '''Function to convert FWHM (Gamma) to standard deviation (sigma)'''
    return Gamma * np.sqrt(2.) / ( np.sqrt(2. * np.log(2.)) * 2. )

def gaussian(x,mu,sigma):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))

wl_ref = np.linspace(360,2550,10000)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
rho_int=[]
for mu,fwhm in prisma_rsr.iterrows():
    sig = Gamma2sigma(fwhm.values)
    rsr = gaussian(wl_ref,mu,sig)
    #rho_ = np.trapz(rho * rsr, wl_ref)/np.trapz(rsr, wl_ref)
    #rho_int.append(rho_)
    axs.plot(wl_ref,rsr,'-k',lw=0.5,alpha=0.4)


# ## Convolution of the transmittance with sensor spectral response

# In[20]:


def get_pressure(alt, psl):
    '''Compute the pressure for a given altitude
       alt : altitude in meters (float or np.array)
       psl : pressure at sea level in hPa
       palt : pressure at the given altitude in hPa'''

    palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
    return palt


# In[21]:


psl=1013
alt=60
pressure=get_pressure(alt, psl)


# In[22]:


sza=dc_l2c.sza.mean().values
vza=dc_l2c.vza.mean().values
raa = dc_l2c.raa.mean().values
M=1./np.cos(np.radians(sza))+1./np.cos(np.radians(vza))


# In[23]:


# Gaseous transmittance
wl_ref = gas_lut.wl#.values
ot_o3 = gas_lut.o3 * to3c
ot_ch4 = gas_lut.ch4 * tch4c
ot_no2 = gas_lut.no2 * tno2c
ot_air = (gas_lut.co+coef_abs_scat*gas_lut.co2+coef_abs_scat*gas_lut.o2+coef_abs_scat*gas_lut.o4)* pressure/1000
ot_other = ot_ch4+ot_no2+ot_o3+ot_air 

Tg = np.exp(-M*ot_other)

Tg_int=[]
for mu,fwhm in prisma_rsr.iterrows():
    sig = Gamma2sigma(fwhm.values)
    rsr = gaussian(wl_ref,mu,sig)
    Tg_ = (Tg * rsr).integrate('wl')/np.trapz(rsr, wl_ref)
    Tg_int.append(Tg_.values)
    
Tg_other = xr.DataArray(Tg_int,name='Ttot',coords={'wl':dc_l1c.wl.values})


# In[24]:


# Water vapor transmittance
Twv=[]
for tcwv in [0,1,5,10,15,20,25,30,40,50,60]:
    ot_wv = gas_lut.h2o *tcwv
    Ttot = np.exp(-M*ot_wv)
    Ttot_int=[]  
    for mu,fwhm in prisma_rsr.iterrows():
        sig = Gamma2sigma(fwhm.values)
        rsr = gaussian(wl_ref,mu,sig)

        Ttot_ = (Ttot * rsr).integrate('wl')/np.trapz(rsr, wl_ref)
        Ttot_int.append(Ttot_.values)
        
    Twv.append(xr.DataArray(Ttot_int,name='Twv',coords={'wl':dc_l1c.wl.values}).assign_coords({'tcwv':tcwv}))    
Twv = xr.concat(Twv,dim='tcwv')


# In[25]:


get_ipython().run_cell_magic('time', '', "Twv_hires = Twv.interp(tcwv=np.linspace(0,60,180),method='cubic' )\n")


# ## Construct coarsened resolution images

# In[26]:


xgeom = dc_l1c[['sza','vza','raa']]
xgeom_mean = xgeom.coarsen(x=xcoarsen, y=xcoarsen).mean()
coarsen_obj=dc_l1c['Rtoa'].coarsen(x=xcoarsen, y=xcoarsen)
Rtoa_mean =coarsen_obj.mean()#.sel(wl=wl_atmo)
masked_mean=masked['Rtoa'].coarsen(x=xcoarsen, y=xcoarsen).mean()
# get number of valid pixels within megapixel
Rtoa_count = masked['Rtoa'].isel(wl=slice(10,20)).mean(dim='wl').coarsen(x=xcoarsen, y=xcoarsen).count()


xc,yc = Rtoa_mean.x,Rtoa_mean.y
#Rtoa_median =coarsen_obj.median()
#Rtoa_std = coarsen_obj.std()


# In[27]:


Rtoa_count.plot.imshow()


# ## Correction for gas absorption

# In[28]:


# correction other gases than water vapor
Rtoa_l2 = Rtoa_mean/Tg_other
masked_l2 = masked_mean/Tg_other


# ## Retrieval of "ad hoc" water vapor transmittance 

# In[29]:


from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes
import itertools
from scipy.optimize import least_squares


def toa_simu(wl,Twv,tcwv,a,b):
    '''wl in micron
    '''
    #print(Twv.tcwv)
    return Twv.interp(tcwv=tcwv,method='linear').values * (a*wl+b)

def toa_simu2(wl,Twv,tcwv,c0,c1,c2,c3):
    
    return c0*np.exp(-c1*wl**-c2) * Twv_.interp(tcwv=tcwv).values +c3 * wl_**-3 *Twv_.interp(tcwv=0.3*tcwv).values 

def fun(x, Twv, wl, y):
    return toa_simu(wl,Twv,*x) - y


def fun2(x, Twv,wl, y):
    return toa_simu2(wl,Twv,*x) - y


# In[30]:


pix_max = xcoarsen*ycoarsen
pix_thresh = 0 #pix_max/2
data= Rtoa_l2.sel(wl=wl_water_vapor)
Twv_ = Twv_hires.sel(wl=wl_water_vapor)
Twv_['wl']=Twv_['wl']/1000
wl_mic = Twv_.wl.values


width,height,nwl = data.shape
result = np.ctypeslib.as_ctypes(np.full((width,height,6),np.nan))
shared_array = sharedctypes.RawArray(result._type_, result)
x0 =[20,-0.04,0.1]


# In[31]:


#Error of parameters
def errFit(hess_inv, resVariance):
    return np.sqrt(np.diag( hess_inv * resVariance))


def chunk_process(args):
    window_x, window_y = args
    tmp = np.ctypeslib.as_array(shared_array)
    x0 =[20,-0.04,0.1]
    for ix in range(window_x, min(width,window_x + block_size)):
        for iy in range(window_y, min(height,window_y + block_size)):
            if Rtoa_count.isel(x=ix,y=iy).values<pix_thresh:
                continue
            y=data.isel(x=ix,y=iy).dropna(dim='wl')
            #sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')
            res_lsq = least_squares(fun, x0, args=(Twv_,wl_mic, y),bounds=([0,-10,0], [60,1,1]),diff_step=1e-2,xtol=1e-2,ftol=1e-2,max_nfev=20)
            x0 = res_lsq.x
            resVariance = (res_lsq.fun**2).sum()/(len(res_lsq.fun)-len(res_lsq.x) ) 
            hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
            try:
                hess_inv = np.linalg.inv(hess)
                std = errFit(hess_inv, resVariance)
            except:
                std=[np.nan,np.nan,np.nan]
            tmp[ix,iy,:] = [*x0,*std]
     
          
window_idxs = [(i, j) for i, j in
               itertools.product(range(0, width, block_size),
                                 range(0, height, block_size))]

p = Pool()
res = p.map(chunk_process, window_idxs)
result = np.ctypeslib.as_array(shared_array)


# In[32]:


gas_img = xr.Dataset(dict(tcwv=(["y", "x"], result[:,:,0].T),
                          tcwv_std=(["y", "x"], result[:,:,3].T)),
            
                     coords=dict(
                          x=xc,
                          y=yc),

                     attrs=dict(
                          description="Fitted Total Columnar Water vapor; warning for transmittance computation only",
                          units="kg/m**2")
                     )


# In[33]:


#fig,axs = plt.subplots(1,6,figsize=(30,4))
#axs=axs.ravel()
#for i in range(6):
#    im=axs[i].imshow(result[...,i])
#    fig.colorbar(im, ax=axs[i],shrink=0.75)


# In[34]:


fig,axs = plt.subplots(1,2,figsize=(11,4))
axs=axs.ravel()
gas_img.tcwv.plot.imshow(cmap=plt.cm.Spectral_r, robust=True,
                               cbar_kwargs={'shrink': shrink,'label':'$tcwv\ (kg\cdot m^{-2})$'},ax=axs[0]) # extent=extent_val, transform=proj, 
gas_img.tcwv_std.plot.imshow(cmap=plt.cm.Spectral_r, robust=True,
                               cbar_kwargs={'shrink': shrink,'label':'$\sigma_{tcwv}\ (kg\cdot m^{-2})$'},ax=axs[1])
for i in range(2):
    axs[i].set(xticks=[], yticks=[])
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')    
    axs[i].set_title(param)                     


# In[35]:


tcwv_vals = gas_img.tcwv.round(1)
tcwvs = np.unique(tcwv_vals)
Twvs = Twv.interp(tcwv=tcwvs,method='linear')
Twv_img = Twvs.interp(tcwv=tcwv_vals,method='nearest')


# In[36]:


Rtoa_l2 = Rtoa_l2/Twv_img


# In[37]:


masked_l2 =masked_l2/Twv_img


# In[38]:


def remove_wl(xarr,wl_to_remove):
    for wls in wl_to_remove:
        wl_min,wl_max=wls
        xarr = xarr.where((xarr.wl < wl_min) | (xarr.wl > wl_max))
    xarr  = xarr.where((masked.wl < 2450))
    return xarr
Rtoa_l2=remove_wl(Rtoa_l2,wl_to_remove)
masked_l2 =remove_wl(masked_l2,wl_to_remove)


# In[39]:


azi=180-raa
from hgrs import metadata

auxdata = metadata()#wl=masked.wl)

wl = Rtoa_l2.wl

sunglint_eps =  auxdata.sunglint_eps['mean'].interp(wl=wl)
#sunglint_eps =sunglint_eps / sunglint_eps.sel(wl=wl_glint).mean(dim='wl')
rot = auxdata.rot.interp(wl=wl)*pressure/pressure_rot_ref

aot_refs=np.logspace(-3,np.log10(1.5),100)
model='MACL_rh70'
#model='DESE_rh70'
aot_hires=aero_lut.sel(model=model).aot.interp(wl=wl/1000,method='quadratic').interp(aot_ref=aot_refs,method='quadratic')#sel(wl=wl_glint)
Rtoa_lut_hires = aero_lut.sel(model=model).I.sel(sza=sza,vza=vza,azi=azi,method='nearest').squeeze().interp(wl=wl/1000,method='quadratic').interp(aot_ref=aot_refs,method='quadratic')/np.cos(np.radians(sza))


# In[40]:


data= masked_l2
pix_thresh = pix_max/2
wl_ = data.wl.values
aot_ = aot_hires.sel(wl=wl_atmo)
rot_ = rot.sel(wl=wl_atmo)
Rtoa_lut_ =Rtoa_lut_hires.sel(wl=wl_atmo)
sunglint_eps_ =sunglint_eps.sel(wl=wl_atmo)

width,height,nwl = data.shape
result = np.ctypeslib.as_ctypes(np.full((width,height,4),np.nan))
shared_array = sharedctypes.RawArray(result._type_, result)


# In[41]:


def transmittance_dir(aot,M,rot=0):
    
    return np.exp(-(rot+aot)*M)

def toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,BRDFg):
    '''wl in micron
    '''
    aot=aot.interp(aot_ref=aot_ref)
    Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
    Tdir = transmittance_dir(aot,M,rot=rot)
    sunglint_corr = Tdir * sunglint_eps
    Rdir=sunglint_corr * BRDFg/Tdir.sel(wl=wl_glint).mean(dim='wl')
    #sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])
    
    return Rdiff+Rdir


def fun(x, wl,aot,rot,Rtoa_lut,sunglint_eps, y):
    return (toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,*x) - y)#/sigma

#Error of parameters
def errFit(hess_inv, resVariance):
    return np.sqrt(np.diag( hess_inv * resVariance))


def chunk_process(args):
    window_x, window_y = args
    tmp = np.ctypeslib.as_array(shared_array)
    x0 =[0.2,0]
    for ix in range(window_x, min(width,window_x + block_size)):
        for iy in range(window_y, min(height,window_y + block_size)):
            if Rtoa_count.isel(x=ix,y=iy).values<pix_thresh:
                continue
            x0 =[0.2,0.002]
            y=data.isel(x=ix,y=iy).dropna(dim='wl')
            #sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')
            res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_, y),bounds=([0.002, 0], [1.45,1.3]),diff_step=1e-6,xtol=1e-2,ftol=1e-2,max_nfev=20)
            x0 = res_lsq.x
            resVariance = (res_lsq.fun**2).sum()/(len(res_lsq.fun)-len(res_lsq.x) ) 
            hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
            
            try:
                hess_inv = np.linalg.inv(hess)
                std = errFit(hess_inv, resVariance)
            except:
                std=[np.nan,np.nan]
            tmp[ix,iy,:] = [*x0,*std]
          
window_idxs = [(i, j) for i, j in
               itertools.product(range(0, width, block_size),
                                 range(0, height, block_size))]

p = Pool()
res = p.map(chunk_process, window_idxs)
result = np.ctypeslib.as_array(shared_array)


# In[42]:


aero_img = xr.Dataset(dict(aot_ref=(["y", "x"], result[:,:,0].T),
                           brdfg=(["y", "x"], result[:,:,1].T),
                           aot_ref_std=(["y", "x"], result[:,:,2].T),
                           brdfg_std=(["y", "x"], result[:,:,3].T)                           
                          ),            
                     coords=dict(
                          x=xc,
                          y=yc),
                     attrs=dict(
                          description="aerosol and sunglint retrieval from corase resoltion data",
                          )
                     )


# In[43]:


aot_ref_vals = aero_img.aot_ref.round(3)
aot_refs = np.unique(aot_ref_vals)
aot_refs= aot_refs[~np.isnan(aot_refs)]


aots = aot_hires.interp(aot_ref=aot_refs,method='linear')
aot_img = aots.interp(aot_ref=aot_ref_vals,method='nearest')
Rdiffs = Rtoa_lut_hires.interp(aot_ref=aot_refs,method='linear')
Rdiff_img = Rdiffs.interp(aot_ref=aot_ref_vals,method='nearest')
Tdir = transmittance_dir(aot_img,M,rot=rot)


# In[44]:


Rcorr = (masked_l2 - Rdiff_img)
Rdir = Tdir * sunglint_eps* (Rcorr.sel(wl=wl_glint)/(Tdir.sel(wl=wl_glint)*sunglint_eps.sel(wl=wl_glint))).mean(dim='wl') 
Rrs_l2 = (Rcorr-Rdir)/np.pi
Rrs_l2.name='Rrs'


# In[55]:


masked


# ## Last step apply atmospheric retrieval to full resolution image

# In[64]:


Tg_tot=(Twv_img*Tg_other).interp(x=masked.x,y=masked.y)


# In[56]:


atmo_img=(Rdir+ Rdiff_img).interp(x=masked.x,y=masked.y)


# In[65]:


L2grs = (masked.Rtoa/Tg_tot-atmo_img)/np.pi


# In[66]:


L2grs.name='Rrs'


# In[45]:


get_ipython().run_line_magic('matplotlib', 'widget')
plt.figure(figsize=(10,5))
ycenter=20
for i in range(3):
    for j in range(3):
        Rdiff = Rdiff_img.isel(x=25+i,y=ycenter+j)
        (Rdiff+Rdir.isel(x=25+i,y=ycenter+j)).plot(x='wl',color='red',alpha=0.5,lw=1.2)#,label='PRISMA')
        Rdiff.plot(x='wl',color='red',alpha=0.5,ls=':',lw=0.7)#,label='PRISMA')
        Rtoa_l2.isel(x=25+i,y=ycenter+j).plot(x='wl',color='black',alpha=0.5,lw=1.2)#
        
plt.hlines(0,350,2500,ls=':',lw=0.5,color='black',zorder=0)
plt.ylabel('$R_{diff}$')
plt.legend()
#plt.xlim(350,500)


# In[46]:


wl_ = Rrs_l2.wl.values
aot_ = aot_hires.sel(wl=wl_)
rot_ = rot.sel(wl=wl_)
Rtoa_lut_ =Rtoa_lut_hires.sel(wl=wl_atmo)
sunglint_eps_ =sunglint_eps.sel(wl=wl_atmo)
Tdir = transmittance_dir(aot_,M,rot=rot)


# In[47]:


params=['aot_ref','aot_ref_std','brdfg','brdfg_std']

fig,axs = plt.subplots(1,4,figsize=(22,4))
axs=axs.ravel()

for i in range(4):
    aero_img[params[i]].plot.imshow(cmap=plt.cm.Spectral_r, robust=True,
                               cbar_kwargs={'shrink': shrink,'label':params[i]},ax=axs[i]) # extent=extent_val, transform=proj, 
    axs[i].set(xticks=[], yticks=[])
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')    
    axs[i].set_title(param)     


# In[ ]:





# In[48]:


fig =Rrs_l2.sel(wl=[440,500,550,600,650,700,750,800,850],method='nearest').plot.imshow(col='wl',col_wrap=3,vmin=0,robust=True,cmap=plt.cm.Spectral_r)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig


# In[49]:


fig =Rtoa_l2.sel(wl=[705,715,805,980,1300,2200],method='nearest').plot.imshow(col='wl',col_wrap=3,robust=True,cmap=plt.cm.Spectral_r)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig


# In[ ]:





# In[67]:


from holoviews import streams
import holoviews as hv
import panel as pn
import param
import numpy as np
import xarray as xr
hv.extension('bokeh')
from holoviews import opts

opts.defaults(
    opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    opts.Image(cmap='binary_r', width=800, height=700),
    opts.Labels(text_color='white', text_font_size='8pt', text_align='left', text_baseline='bottom'),
    opts.Path(color='white'),
    opts.Spread(width=900),
    opts.Overlay(show_legend=True))
# set the parameter for spectra extraction
hv.extension('bokeh')
pn.extension()


param = 'Rrs' #Rtoa'
#img = dc_l1c[['Rtoa','Ltoa']] 
raster = L2grs #masked[param] 
#img = dc_l1c[['Rtoa','Ltoa']] 
vmax = 0.04
#param = 'rho'
#raster = dc_l2c[param] 
cmap='RdBu_r'
cmap='Spectral_r'
third_dim = 'wl'

wl= raster.wl.data
Nwl = len(wl)
ds = hv.Dataset(raster.persist())
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap=cmap ,colorbar=True,clim=(0.00,vmax)).hist(bin_range=(0,0.2)) 

polys = hv.Polygons([])
box_stream = hv.streams.BoxEdit(source=polys)
dmap, dmap_std=[],[]

def roi_curves(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]= hv.Curve((wl,mean[param]),'Wavelength (nm)', param) 

    return hv.NdOverlay(curves)


# a bit dirty to have two similar function, but holoviews does not like mixing Curve and Spread for the same stream
def roi_spreads(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]=  hv.Spread((wl,mean[param],std[param]),fill_alpha=0.3)

    return hv.NdOverlay(curves)

mean=hv.DynamicMap(roi_curves,streams=[box_stream])
std =hv.DynamicMap(roi_spreads, streams=[box_stream])    
hlines = hv.HoloMap({wl[i]: hv.VLine(wl[i]) for i in range(Nwl)},third_dim )


hv.output(widget_location='top_left')

# visualize and play
graphs = ((mean* std *hlines).relabel(param))
layout = (im * polys +graphs    ).opts(opts.Image(tools=['hover']),
    opts.Curve(width=750,height=500, framewise=True,xlim=(400,2500),tools=['hover']), 
    opts.Polygons(fill_alpha=0.2, color='green',line_color='black'), 
    opts.VLine(color='black')).cols(2)
layout 


# In[ ]:





# In[68]:


from aeronet_visu import data_loading as dl
dir = '/DATA/AERONET/OCv3/'
figdir= '/DATA/AERONET/fig'
aeronet_site = 'Bahia_Blanca'
file = aeronet_site+'_OCv3.lev15'

irr = dl.irradiance()
irr.load_F0()

param = 'Lwn'

# ---------------------------------------------
# Load data and convert into xarray
# ---------------------------------------------

df = dl.read(opj(dir, file)).read_aeronet_ocv3()

df = df.droplevel(0, 1)
# criteria to select scalar and spectrum values
criteria = df.columns.get_level_values(1) == ''
df_att = df.loc[:, criteria].droplevel(1, 1)
df_spec = df.loc[:, ~criteria]

ds = df_spec.stack().to_xarray()
ds = xr.merge([ds, df_att.to_xarray()])
del df

ds = ds.assign_coords({'level_1': ds.level_1.astype(float)}).rename({'level_1': "wl"})
ds['SZA'] = ds.Solar_Zenith_Angle.mean(axis=1)
ds['year']=ds['date.year']
ds['season']=ds['date.season']

wl = ds.wavelength * 1000
ds['Rrs'] = ds[param] / (irr.get_F0(wl) * 0.1)


# In[69]:


plt.figure(figsize=(10,5))
xcenter,ycenter=500,500
for i in range(5):
    for j in range(5):
        L2grs.isel(x=xcenter+i,y=ycenter+j).plot(x='wl',color='black',alpha=0.5,lw=1.2)#,label='PRISMA')
ds.Rrs.sel(date='2021-08-14',method='nearest').plot(x='wl',marker='o',color='red',label='AERONET-OC')
plt.hlines(0,350,2500,ls=':',lw=0.5,color='black',zorder=0)
plt.ylabel('$R_{rs}\ (sr^{-1})$')
plt.legend()
#plt.xlim(390,1100)


# In[41]:


from scipy.interpolate import RegularGridInterpolator
x,y= range(width),range(height)
interp = RegularGridInterpolator((x, y), res,
                                 bounds_error=False, fill_value=None)
xx,yy = np.linspace(0,width,len(dc_l1c.x)),np.linspace(0,height,len(dc_l1c.y)),
X, Y = np.meshgrid(xx, yy, indexing='ij')
tcwv_img = xr.DataArray(data=interp((X,Y)),dims=["y", "x"],
                        coords=dict(
                            x=dc_l1c.x.values,
                            y=dc_l1c.y.values),
                        attrs=dict(
                            description="Fitted Total Columnar Water vapor; warning for transmittance computation only",
                            units="kg/m**2")
                        )


# In[ ]:


get_ipython().run_cell_magic('time', '', "tcwvs = np.unique(tcwv_img.round(1))\nTwvs = Twv.interp(tcwv=tcwvs,method='linear')\nTwv_img = Twvs.interp(tcwv=tcwv_img.round(1),method='nearest')\nTwv_img\n")


# In[43]:


#%%time
#Twv_img2 = Twv.interp(tcwv=tcwv_img,method='linear')
#Twv_img2


# In[44]:


#adiff=np.abs(Twv_img-Twv_img2 )


# In[45]:


#print('absolute transmittance error\n max={:.5f}\n mean={:.5f}\n median={:.5f}\n quantile90={:.5f}'.\
#      format(np.max(adiff).values,np.mean(adiff).values,np.median(adiff),np.quantile(adiff,0.9)))


# In[46]:


coarsening=1
gamma=0.7
brightness_factor=1.5
fig,axs = plt.subplots(2,2,figsize=(15,12))
axs=axs.ravel()
im= axs[0].imshow(res[:,::-1],cmap=plt.cm.YlGnBu)
im2=axs[1].imshow(tcwv_img.values[:,::-1],cmap=plt.cm.YlGnBu)#,add_colorbar=False)#,robust=True,)
im3=axs[2].imshow(Twv_img.sel(wl=900,method='nearest').values[:,::-1],cmap=plt.cm.Spectral_r)
p = (dc_l1c.Rtoa[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])**gamma*brightness_factor).plot.imshow(rgb='wl',robust=True,ax=axs[3])#, subplot_kws=dict(projection= l1c.proj))

fig.colorbar(im, ax=axs[0],shrink=0.75, label='$tcwv\ (kg\cdot m^{-2})$')
fig.colorbar(im2, ax=axs[1],shrink=0.75, label='$tcwv\ (kg\cdot m^{-2})$')
fig.colorbar(im3, ax=axs[2],shrink=0.75, label='$Transmittance$')



#  ## Application to satellite data

# In[148]:


to3c=6.5e-3
tno2c=3e-6
tch4c= 1e-2
pressure=1013
coef_abs_scat=0.3

#['ch4','co','co2','h2o','n2o','no2','o2','o3','o4']


ot_o3 = ot.o3 * to3c
ot_ch4 = ot.ch4 * tch4c
ot_no2 = ot.no2 * tno2c
ot_others = (ot.co+coef_abs_scat*ot.co2+ot.n2o+coef_abs_scat*ot.o2+coef_abs_scat*ot.o4)* pressure/1000

ot_tot = ot_ch4+ot_no2+ot_o3+ot_others 


M=1./np.cos(np.radians(sza))+1./np.cos(np.radians(vza))
Tg = np.exp(-M*ot_tot)
wl_ref = ot_tot.wl
Tg_int=[]
for mu,fwhm in prisma_rsr.iterrows():
    sig = Gamma2sigma(fwhm.values)
    rsr = gaussian(wl_ref,mu,sig)

    Tg_ = (Tg * rsr).integrate('wl')/np.trapz(rsr, wl_ref)
    Tg_int.append(Tg_.values)
Tg_sat = xr.DataArray(Tg_int,name='Ttot',coords={'wl':dc_l1c.wl.values})


# In[48]:


Ttot_sat = Tg_sat*Twv_img
masked['Rtoa'] = masked.Rtoa / Ttot_sat 


# In[ ]:





# In[49]:


masked['Rtoa'] = masked['Rtoa'].where(Ttot_sat > 0.1)


# In[50]:


wl_to_remove = [(935,967),(1105,1170),(1320,1490),(1778,2033)]
for wls in wl_to_remove:
    wl_min,wl_max=wls
    masked['Rtoa'] = masked['Rtoa'].where((masked.wl < wl_min) | (masked.wl > wl_max))
masked['Rtoa'] = masked['Rtoa'].where((masked.wl < 2450))


# In[51]:


x,y=600,600
pixel = dc_l1c.isel(x=x,y=y)
tcwv = tcwv_img.isel(x=x,y=y)
lw=0.5
fig,axs = plt.subplots(nrows=1,figsize=(15,5))
axs.minorticks_on()
pixel.Rtoa.plot(lw=lw,color='black',ax=axs)
(pixel.Rtoa/Tg_sat/Twv.interp(tcwv=tcwv)).plot(lw=0.8,color='red',ax=axs)
(pixel.Rtoa/Tg_sat).plot(lw=lw,color='grey',ax=axs)
masked['Rtoa'].isel(x=x,y=y).plot(ls=':',color='blue',ax=axs)
plt.ylim(0,0.2)


# In[52]:


#param = 'Rtoa'
#img = dc_l1c[['Rtoa','Ltoa']] 
#raster = img[param]  

#param = 'rho'
#img = dc_l2c[param] 
#raster = dc_l1c[param] 
#raster = img#[param]  


# In[55]:


azi=180-raa

Rtoa_lut = lut.I.sel(sza=sza,vza=vza,azi=azi,method='nearest').sel(model='COPO_rh99').squeeze()/np.cos(np.radians(sza))
Rtoa_lut['wl']= Rtoa_lut['wl']*1000


# In[56]:


from hgrs import metadata


wl_glint=slice(2100,2350)
wl_atmo = slice(950,2500)


auxdata = metadata()#wl=masked.wl)
wlref=2200
wl = masked.wl

sunglint_eps =  auxdata.sunglint_eps['mean'].interp(wl=wl)
sunglint_eps =sunglint_eps / sunglint_eps.sel(wl=wl_glint).mean(dim='wl')
rot = auxdata.rot.interp(wl=wl)


# In[57]:


plt.figure()
lut.sel(model='COPO_rh99').aot.interp(wl=masked.wl/1000,method='cubic').plot(hue='aot_ref')
aot=lut.sel(model='COPO_rh99').aot.interp(aot_ref=0.2).interp(wl=masked.wl/1000,method='cubic')
ssa=lut.sel(model='COPO_rh99').ssa.isel(aot_ref=0).interp(wl=masked.wl/1000,method='cubic')


# In[58]:


M = 1/np.cos(np.radians(sza)) + 1/np.cos(np.radians(vza))
def transmittance_dir(aot,M,rot=0):
    
    return np.exp(-(rot+aot)*M)
plt.figure()               

Tdir = transmittance_dir(aot,M)
Tdir.plot()
Tdir = transmittance_dir((1-ssa)*aot,M)
Tdir.plot()
Tdir = transmittance_dir(aot,M,rot=rot)
Tdir.plot()


# In[175]:


data=masked.Rtoa
#data=Rtoa_l2

cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                                                    ['navy', "blue", 'lightskyblue',
                                                     "grey",   'forestgreen','yellowgreen',
                                                     "khaki", "gold",
                                                     'orangered', "firebrick", 'purple'])

norm = mpl.colors.Normalize(vmin=0.01,vmax=1.1)#-3, vmax=np.log10(1.5))

models = ['COAV', 'DESE', 'MACL', 'MAPO', 'COPO', 'URBA']
rh='_rh70'

fig,axs = plt.subplots(2,3,figsize=(28,11),sharey=True,sharex=True)
axs=axs.ravel()
for i_,model_ in  enumerate(models):
    model=model_+rh
    for i in range(5):
        for j in range(5):
            pixel=data.sel(x=499+i,y=631+j)
            pixel.plot(x='wl',color='black',alpha=0.5,ax=axs[i_])#,label='PRISMA')
    #pixel.Rtoa.plot(ax=axs[i_])
    Rtoa_lut = aero_lut.I.sel(model=model).sel(sza=sza,vza=vza,azi=azi,method='nearest').squeeze()/np.cos(np.radians(sza))
    Rtoa_lut['wl']= Rtoa_lut['wl']*1000
    Rtoa_lut=Rtoa_lut.interp(wl=masked.wl.values,method='cubic')
    aot=aero_lut.sel(model=model).aot.interp(wl=masked.wl/1000,method='cubic')
    Tdir = transmittance_dir(aot,M,rot=rot)
    sunglint_corr = Tdir * sunglint_eps
    sunglint_toa=sunglint_corr *0.1#( pixel.sel(wl=slice(2150,2350))-Rtoa_lut.sel(wl=slice(2150,2350))).mean(dim='wl')
    #sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])
    
    Rsim = ( Rtoa_lut+sunglint_toa)
    for aot_ref in Rtoa_lut.aot_ref.values[1:]:
        Rtoa_lut.sel(aot_ref=aot_ref).plot(x='wl',color=cmap(norm(aot_ref)),ls=':',lw=1,ax=axs[i_])
        Rsim.sel(aot_ref=aot_ref).plot(x='wl',color=cmap(norm(aot_ref)),label=str(aot_ref),lw=1,ax=axs[i_])
    Rtoa_sim= Rsim.interp(aot_ref=0.4)
    #Rtoa_sim.plot(x='wl',color='black',ls='--',ax=axs[i_])
    axs[i_].set_title(model)
    axs[i_].minorticks_on()
    axs[i_].legend()


# In[249]:


def toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,BRDFg):
    '''wl in micron
    '''
    aot=aot.interp(aot_ref=aot_ref)
    Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
    Tdir = transmittance_dir(aot,M,rot=rot)
    sunglint_corr = Tdir * sunglint_eps
    Rdir=sunglint_corr * BRDFg
    #sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])
    
    return Rdiff+Rdir


def fun(x, wl,aot,rot,Rtoa_lut,sunglint_eps, sigma,y):
    return (toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,*x) - y)/sigma

wl_glint=slice(2100,2350)
wl_atmo = slice(950,2500)

aot=lut.sel(model=model).aot.interp(wl=wl/1000,method='cubic')#.sel(wl=wl_glint)

model='MAPO_rh70'
aot=lut.sel(model=model).aot.interp(wl=wl/1000,method='cubic')#.sel(wl=wl_glint)
Rtoa_lut = lut.I.sel(model=model).sel(sza=sza,vza=vza,azi=azi,method='nearest').squeeze()/np.cos(np.radians(sza))
Rtoa_lut['wl']= Rtoa_lut['wl']*1000
Rtoa_lut=Rtoa_lut.interp(wl=masked.wl,method='cubic')

aot_ref=0.45
BRDFg=1e-3
plt.figure()
for i in range(5):
    for j in range(5):
        pixel=masked.Rtoa.sel(x=499+i,y=631+j)
        pixel.plot(x='wl',color='black',alpha=0.5)#,label='PRISMA')
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,BRDFg).plot()
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,0.01,BRDFg).plot(label='clear')
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,1e-2).plot()
plt.legend()


# ## Coarsen resolution for aerosol retrieval

# In[230]:


xcoarsen,ycoarsen=25,25
xgeom = masked[['sza','vza','raa']]
xgeom_mean = xgeom.coarsen(x=xcoarsen, y=xcoarsen).mean()
xRtoa = masked['Rtoa'].sel(wl=wl_atmo)
xgeom_mean


# In[227]:


coarsen_obj=xRtoa.coarsen(x=xcoarsen, y=xcoarsen)
Rtoa_mean =coarsen_obj.mean()#.sel(wl=wl_atmo)
Rtoa_median =coarsen_obj.median()
Rtoa_std = coarsen_obj.std()
Rtoa_count = xRtoa.isel(wl=20).coarsen(x=xcoarsen, y=xcoarsen).count()


# In[232]:


Rtoa_count = xRtoa.mean(dim='wl').coarsen(x=xcoarsen, y=xcoarsen).count()
plt.figure()
Rtoa_count.plot.imshow()


# In[184]:


y=img.isel(x=20,y=25).sel(wl=wl_atmo).dropna(dim='wl')

wl_ = y.wl.values
aot_ = aot.sel(wl=wl_atmo)
rot_ = rot.sel(wl=wl_atmo)
Rtoa_lut_ =Rtoa_lut.sel(wl=wl_atmo)
sunglint_eps_ =sunglint_eps.sel(wl=wl_atmo)
x0=[0.4,0]
res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_, y),bounds=([0.002, 0], [1.45,3]))#,diff_step=1e-3,xtol=1e-2,ftol=1e-2,max_nfev=10)
x0 = res_lsq.x
x0


# In[186]:


plt.figure()
img.isel(x=20,y=25).plot()
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,*x0).plot()


# In[237]:


from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes
import itertools

block_size = 2

data= Rtoa_mean

wl_ = data.wl.values
aot_ = aot.sel(wl=wl_atmo)
rot_ = rot.sel(wl=wl_atmo)
Rtoa_lut_ =Rtoa_lut.sel(wl=wl_atmo)
sunglint_eps_ =sunglint_eps.sel(wl=wl_atmo)

width,height,nwl = data.shape
result = np.ctypeslib.as_ctypes(np.full((width,height,4),np.nan))
shared_array = sharedctypes.RawArray(result._type_, result)


# In[250]:


def toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,BRDFg):
    '''wl in micron
    '''
    aot=aot.interp(aot_ref=aot_ref)
    Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
    Tdir = transmittance_dir(aot,M,rot=rot)
    sunglint_corr = Tdir * sunglint_eps
    Rdir=sunglint_corr * BRDFg
    #sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])
    
    return Rdiff+Rdir


def fun(x, wl,aot,rot,Rtoa_lut,sunglint_eps, sigma,y):
    return (toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,*x) - y)/sigma

#Error of parameters
def errFit(hess_inv, resVariance):
    return np.sqrt(np.diag( hess_inv * resVariance))


def chunk_process(args):
    window_x, window_y = args
    tmp = np.ctypeslib.as_array(shared_array)
    x0 =[0.2,0]
    for ix in range(window_x, min(width,window_x + block_size)):
        for iy in range(window_y, min(height,window_y + block_size)):
            if Rtoa_count.isel(x=ix,y=iy).values<300:
                continue
            y=Rtoa_mean.isel(x=ix,y=iy).dropna(dim='wl')
            #sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')
            res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_,sigma, y),bounds=([0.002, 0], [1.45,3]),diff_step=1e-6,xtol=1e-2,ftol=1e-2,max_nfev=20)
            x0 = res_lsq.x
            resVariance = (res_lsq.fun**2).sum()/(len(res_lsq.fun)-len(res_lsq.x) ) 
            hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
            hess_inv = np.linalg.inv(hess)
            std = errFit(hess_inv, resVariance)
            tmp[ix,iy,:] = [*x0,*std]
     
          
window_idxs = [(i, j) for i, j in
               itertools.product(range(0, width, block_size),
                                 range(0, height, block_size))]

p = Pool()
res = p.map(chunk_process, window_idxs)
result = np.ctypeslib.as_array(shared_array)


# In[252]:


title=['aot','brdfg','aot_sd','brdfg_sd']
vmax=[0.2,0.01,0.05,0.001]
fig,axs = plt.subplots(2,2,figsize=(15,12))
axs=axs.ravel()
for i in range(4):
    im= axs[i].imshow(result[::-1,:,i].T,vmax=vmax[i],cmap=plt.cm.Spectral_r)
    fig.colorbar(im, ax=axs[i],shrink=0.75)
    axs[i].set_title(title[i])



# In[123]:


res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_, y))#,diff_step=1e-3,xtol=1e-2,ftol=1e-2,max_nfev=10)
x0 = res_lsq.x
x0


# In[126]:


get_ipython().run_line_magic('matplotlib', 'widget')
# config
wl_atmo = slice(900,2440)
aot_refs=np.linspace(0.001,1.5,100)
model='MACL_rh99'
aot_hires=lut.sel(model=model).aot.interp(wl=wl/1000,method='cubic').interp(aot_ref=aot_refs,method='cubic')#sel(wl=wl_glint)
Rtoa_lut = lut.I.sel(model=model).sel(sza=sza,vza=vza,azi=azi,method='nearest').squeeze()/np.cos(np.radians(sza))
Rtoa_lut['wl']= Rtoa_lut['wl']*1000
Rtoa_lut=Rtoa_lut.interp(wl=masked.wl,method='cubic')

# data preparation
y=masked.Rtoa.sel(x=499,y=631).sel(wl=wl_atmo).dropna(dim='wl')
wl_ = y.wl.values
aot_ = aot.sel(wl=wl_atmo)
rot_ = rot.sel(wl=wl_atmo)
Rtoa_lut_ =Rtoa_lut.sel(wl=wl_atmo)
sunglint_eps_ =sunglint_eps.sel(wl=wl_atmo)

# Inversion
x0=[0.1,0]
res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_, y),bounds=([0, 0], [1,3]))#,diff_step=1e-3,xtol=1e-2,ftol=1e-2,max_nfev=10)
x0 = res_lsq.x
x0
x0=[0.002,0]
# plotting
Rtoa_sim = toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,*x0)
fig,axs = plt.subplots(1,2,figsize=(16,5),sharex=True)
y.plot(x='wl',color='black',alpha=0.75,lw=2,ax=axs[0])
for i in range(5):
    for j in range(5):
        pixel=masked.Rtoa.sel(x=499+i,y=631+j)
        pixel.plot(x='wl',color='black',alpha=0.5,lw=0.5,ax=axs[0])#,label='PRISMA')
        pixel=masked.Rtoa.sel(x=499+i,y=631+j)-Rtoa_sim
        pixel.plot(x='wl',color='black',alpha=0.5,ax=axs[1])

Rtoa_sim.plot(label='GRS',ax=axs[0])
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,1e-2).plot(ax=axs[0])
axs[0].set_title(x0)
plt.legend()


# In[64]:


w,h=9,9
res = np.full((w,h,2),np.nan)
x0 = np.array([0.2,0])
for i in range(w):
    for j in range(h):
        y=masked.Rtoa.sel(x=499+i,y=631+j).sel(wl=wl_atmo).dropna(dim='wl')
        
        res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_, y),bounds=([0.001, 0], [1,3]), loss='soft_l1',f_scale=0.01,diff_step=1e-3,xtol=1e-2,ftol=1e-2,max_nfev=10)
        x0 = res_lsq.x       
        
        res[i,j,:] = x0


# In[65]:


fig,axs = plt.subplots(1,2,figsize=(16,5))
for i in range(w):
    for j in range(h):
        Rtoa_sim = toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,*res[i,j,:])
        pixel=masked.Rtoa.sel(x=499+i,y=631+j)
        pixel.plot(x='wl',color='black',alpha=0.5,lw=0.5,ax=axs[0])#,label='PRISMA')
        Rrs=(masked.Rtoa.sel(x=499+i,y=631+j)-Rtoa_sim)/np.pi
        Rrs.plot(x='wl',color='black',alpha=0.5,ax=axs[1],lw=0.5)

        Rtoa_sim.plot(label=res[i,j,:],ax=axs[0])

axs[0].legend()


# In[246]:


from aeronet_visu import data_loading as dl
dir = '/DATA/AERONET/OCv3/'
figdir= '/DATA/AERONET/fig'
aeronet_site = 'Bahia_Blanca'
file = aeronet_site+'_OCv3.lev15'

irr = dl.irradiance()
irr.load_F0()

param = 'Lwn'

# ---------------------------------------------
# Load data and convert into xarray
# ---------------------------------------------

df = dl.read(opj(dir, file)).read_aeronet_ocv3()

df = df.droplevel(0, 1)
# criteria to select scalar and spectrum values
criteria = df.columns.get_level_values(1) == ''
df_att = df.loc[:, criteria].droplevel(1, 1)
df_spec = df.loc[:, ~criteria]

ds = df_spec.stack().to_xarray()
ds = xr.merge([ds, df_att.to_xarray()])
del df

ds = ds.assign_coords({'level_1': ds.level_1.astype(float)}).rename({'level_1': "wl"})
ds['SZA'] = ds.Solar_Zenith_Angle.mean(axis=1)
ds['year']=ds['date.year']
ds['season']=ds['date.season']

wl = ds.wavelength * 1000
ds['Rrs'] = ds[param] / (irr.get_F0(wl) * 0.1)


# In[248]:


ds.sel(date='2021-08-14',method='nearest')


# In[247]:


plt.figure(figsize=(10,5))
for i in range(w):
    for j in range(h):
        Rtoa_sim = toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,0.02,0.0041)#res[i,j,1]
        pixel=masked.Rtoa.sel(x=499+i,y=631+j)
        
        Rrs=(masked.Rtoa.sel(x=499+i,y=631+j)-Rtoa_sim)/np.pi
        Rrs.plot(x='wl',color='black',alpha=0.5,lw=0.5)
        
ds.Rrs.sel(date='2021-08-14',method='nearest').plot(x='wl',marker='o',color='red',label='AERONET-OC')
plt.hlines(0,350,2500,ls=':',lw=0.5,color='black',zorder=0)
plt.ylabel('$R_{rs}\ (sr^{-1})$')
plt.legend()
#plt.xlim(350,1100)


# In[68]:


plt.figure()
plt.plot(wl_,res_lsq.fun)


# In[69]:


#Error of parameters
def errFit(hess_inv, resVariance):
    return np.sqrt(np.diag( hess_inv * resVariance))
res_lsq
resVariance = (res_lsq.fun**2).sum()/(len(res_lsq.fun)-len(res_lsq.x) ) 
hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
hess_inv = np.linalg.inv(hess)
errFit(hess_inv, resVariance)


# In[70]:


wl = masked.wl
aot_refs=np.logspace(-3,np.log10(1.5),100)
model='MACL_rh99'
model='DESE_rh70'
aot_hires=lut.sel(model=model).aot.interp(wl=wl/1000,method='quadratic')#.interp(aot_ref=aot_refs,method='cubic')#sel(wl=wl_glint)

Rtoa_lut_hires = lut.sel(model=model).I.sel(sza=sza,vza=vza,azi=azi,method='nearest').squeeze().interp(wl=wl/1000,method='quadratic').interp(aot_ref=aot_refs,method='quadratic')
plt.figure()
Rtoa_lut_hires.isel(wl=10).plot()


# In[71]:


y.shape


# In[72]:


y=masked.Rtoa.sel(x=499,y=[631,632]).sel(wl=wl_atmo).dropna(dim='wl')#.values#wl_ = y.wl.values
wl_=y.wl
ydata = y.data
aot_ = aot_hires.sel(wl=wl_)
rot_ = rot.sel(wl=wl_)
Rtoa_lut_ =Rtoa_lut.sel(wl=wl_)
sunglint_eps_ =sunglint_eps.sel(wl=wl_)
aot_ref=x[0]
aot=aot_.interp(aot_ref=aot_ref)
Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
Tdir = transmittance_dir(aot,M,rot=rot)
sunglint_corr = Tdir * sunglint_eps
print(sunglint_corr)
cost=[]
for i in range(2):
    Rdir=sunglint_corr * x[i+1]
#sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])


    cost.append(Rdiff+Rdir - y[i,:])


# In[160]:


def toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,BRDFg):
    '''wl in micron
    '''
    
    
    
    aot=aot.interp(aot_ref=aot_ref)
    Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
    Tdir = transmittance_dir(aot,M,rot=rot)
    sunglint_corr = Tdir * sunglint_eps
    
    Rdir=sunglint_corr * BRDFg
    #sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])
    
    return Rdiff+Rdir


def fun(x, wl,aot,rot,Rtoa_lut,sunglint_eps, y):
    #print(x)
    aot_ref=x[0]
    aot_=aot.interp(aot_ref=aot_ref).values
    Rdiff = Rtoa_lut.interp(aot_ref=aot_ref).values
   
    Tdir = transmittance_dir(aot_,M,rot=rot)
    sunglint_corr = Tdir * sunglint_eps
    cost=[]
    for i in range(len(x)-1):
        Rdir=sunglint_corr * x[i+1]
         
        cost=np.concatenate([cost, Rdiff+Rdir - y[:,i]])
    #print(np.sum(cost))
    return cost


# In[164]:


wl_glint=slice(2100,2350)
wl_atmo = slice(950,2500)

y=masked.Rtoa.sel(x=slice(505,500),y=slice(635,630)).sel(wl=wl_atmo).dropna(dim='wl').stack(z=("x", "y"))#.values#wl_ = y.wl.values
wl_=y.wl
ydata = y.data
Ndata = ydata.shape[1]
aot_ = aot_hires.sel(wl=wl_)
rot_ = rot.sel(wl=wl_)
Rtoa_lut_ =Rtoa_lut.sel(wl=wl_)
sunglint_eps_ =sunglint_eps.sel(wl=wl_)       
zeros =[0]*Ndata
x0=[0.4,*zeros]
res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_.data,Rtoa_lut_,sunglint_eps_.data, ydata),verbose=True,diff_step=1e-6)#,xtol=1e-2,ftol=1e-2,max_nfev=10)
x0 = res_lsq.x
x0
        


# In[163]:


res_lsq


# In[81]:


plt.figure()
plt.imshow(res_lsq.x[1:].reshape(6,6))


# In[101]:


data=masked.Rtoa.sel(wl=wl_atmo).coarsen(x=5, y=5).construct(x=('xc','xf'),y=('yc','yf')) #.stack(z=("x", "y"))


# In[103]:


data#.dropna(dim='wl')


# In[109]:


#plt.figure()
y = data.isel(yc=110,xc=100).dropna(dim='wl').stack(z=("xf", "yf"))
ydata = y.data
Ndata = ydata.shape[1]
Ndata


# In[108]:





# In[113]:


def errFit(hess_inv, resVariance):
    return np.sqrt(np.diag( hess_inv * resVariance))


aot_ = aot_hires.sel(wl=wl_)
rot_ = rot.sel(wl=wl_)
Rtoa_lut_ =Rtoa_lut.sel(wl=wl_)
sunglint_eps_ =sunglint_eps.sel(wl=wl_)

y = data.isel(yc=110,xc=100).dropna(dim='wl').stack(z=("xf", "yf"))
ydata = y.data
Ndata = ydata.shape[1]
zeros =[0]*Ndata
x0=[0.4,*zeros]

width,height=9,9
res = np.full((width,height,3),np.nan)

for iy in range(height):
    print(iy)
    for ix in range(width):
        y = data.isel(yc=110+iy,xc=100+ix).dropna(dim='wl').stack(z=("xf", "yf"))
        ydata = y.data
        Ndata = ydata.shape[1]
        res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_.data,Rtoa_lut_,sunglint_eps_.data, ydata),diff_step=1e-6,xtol=1e-2,ftol=1e-2,max_nfev=10)
        x0 = res_lsq.x
        resVariance = (res_lsq.fun**2).sum()/(len(res_lsq.fun)-len(res_lsq.x) ) 
        hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
        hess_inv = np.linalg.inv(hess)
        aot_sd = errFit(hess_inv, resVariance)[0]
        res[ix,iy,:] = x0[0],aot_sd,Ndata


# In[117]:


plt.figure()
plt.imshow(res[:,:,0])
plt.colorbar()


# In[ ]:


img = dc_l1c.Rtoa.sel(wl=wl_water_vapor).coarsen(x=25, y=25).mean()
wl_= img.wl.values
wl_mic=wl_/1000

data=masked.Rtoa.sel(wl=wl_atmo).dropna(dim='wl').coarsen(x=5, y=5) #.stack(z=("x", "y"))
data =img.values
width,height,nwl = data.shape
res = np.full((width,height),np.nan)
x0 = np.array([20,-0.04,0.1])
for iy in range(height):
    print(iy)
    for ix in range(width):
        #print(data[ix,iy])
        data.isel(yc=100,xc=100)
        res_lsq = least_squares(fun, x0, args=(wl_,aot_,rot_.data,Rtoa_lut_,sunglint_eps_.data, ydata),diff_step=1e-6,xtol=1e-2,ftol=1e-2,max_nfev=10)   
        x0 = res_lsq.x
        res[ix,iy] = x0[0]
     


# In[80]:


plt.figure()
y=masked.Rtoa.sel(x=slice(505,500),y=slice(635,630)).dropna(dim='wl').stack(z=("x", "y"))#.values#wl_ = y.wl.values
for z,y_ in y.groupby('z'):
    plt.plot(y_.wl,y_.values)


# In[ ]:


ydata
x=[0.0010005 , 0.00533898,0]
fun(x,wl_,aot_,rot_.data,Rtoa_lut_,sunglint_eps_.data, ydata)


# In[ ]:


masked.coarsen(y=9, x=9)#.construct(lon=("x_coarse", "x_fine"), lat=("y_coarse", "y_fine"))




# In[ ]:


def chunk_process(args):
    window_x, window_y = args
    tmp = np.ctypeslib.as_array(shared_array)

    for ix in range(window_x, min(x_,window_x + block_size)):
        for iy in range(window_y, min(y_,window_y + block_size)):
            if np.isnan(Rrs__[0, ix, iy]):
                continue
            Rrs_ = Rrs__[:, ix, iy]
            if (Rrs_ >= -1).all() and (Rrs_ <= 0.21).all():
                print(Rrs_)
                Rrs_[Rrs_<0]=0

                out = algo.call_solver(Rrs, xinit=xinit,xtol=1e-4, ftol=1e-4)
                for ip, (name, param) in enumerate(out.params.items()):
                    tmp[2 * ip, ix, iy] = param.value
                    if param.stderr is not None:
                        tmp[2 * ip + 1, ix, iy] = param.stderr / param.value

                tmp[-1, ix, iy] = out.redchi

window_idxs = [(i, j) for i, j in
               itertools.product(range(0, x_, block_size),
                                 range(0, y_, block_size))]

p = Pool()
res = p.map(chunk_process, window_idxs)
result = np.ctypeslib.as_array(shared_array)


# In[304]:


aot=lut.sel(model=model).aot.interp(wl=wl/1000,method='cubic')#.sel(wl=wl_glint)

model='MAPO_rh70'
Rtoa_lut = lut.I.sel(model=model).sel(sza=sza,vza=vza,azi=azi,method='nearest').squeeze()/np.cos(np.radians(sza))
Rtoa_lut['wl']= Rtoa_lut['wl']*1000
Rtoa_lut=Rtoa_lut.interp(wl=masked.wl,method='cubic')

aot_ref=0.45
BRDFg=1e-3
plt.figure()
for i in range(5):
    for j in range(5):
        pixel=masked.Rtoa.sel(x=499+i,y=631+j)
        pixel.plot(x='wl',color='black',alpha=0.5)#,label='PRISMA')
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,BRDFg).plot()
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,0.1,BRDFg).plot(label='clear')
toa_simu(wl,aot,rot,Rtoa_lut,sunglint_eps,aot_ref,1e-2).plot()
plt.legend()


# In[ ]:


Rrs = (img.Rtoa-Rtoa_sim)/np.pi
Rrs.name='Rrs'


# In[211]:


Rrs_l2.name='Rrs'


# In[212]:


param = 'Rrs' #Rtoa'
#img = dc_l1c[['Rtoa','Ltoa']] 
raster = Rrs_l2 #masked[param]  
#param = 'rho'
#raster = dc_l2c[param] 

third_dim = 'wl'

wl= raster.wl.data
Nwl = len(wl)
ds = hv.Dataset(raster)
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap= 'RdBu_r',colorbar=True,clim=(0.00,0.2035)).hist(bin_range=(0,0.2)) 


# In[114]:


from holoviews import streams
import holoviews as hv
import panel as pn
import param
import numpy as np
import xarray as xr
hv.extension('bokeh')
from holoviews import opts

opts.defaults(
    opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    opts.Image(cmap='binary_r', width=800, height=700),
    opts.Labels(text_color='white', text_font_size='8pt', text_align='left', text_baseline='bottom'),
    opts.Path(color='white'),
    opts.Spread(width=900),
    opts.Overlay(show_legend=True))
# set the parameter for spectra extraction
hv.extension('bokeh')
pn.extension()


param = 'Rrs' #Rtoa'
#img = dc_l1c[['Rtoa','Ltoa']] 
raster = Rrs_l2 #masked[param] 
#img = dc_l1c[['Rtoa','Ltoa']] 
vmax = 0.04
#param = 'rho'
#raster = dc_l2c[param] 
cmap='RdBu_r'
cmap='Spectral_r'
third_dim = 'wl'

wl= raster.wl.data
Nwl = len(wl)
ds = hv.Dataset(raster.persist())
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap=cmap ,colorbar=True,clim=(0.00,vmax)).hist(bin_range=(0,0.2)) 

polys = hv.Polygons([])
box_stream = hv.streams.BoxEdit(source=polys)
dmap, dmap_std=[],[]

def roi_curves(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]= hv.Curve((wl,mean[param]),'Wavelength (nm)', param) 

    return hv.NdOverlay(curves)


# a bit dirty to have two similar function, but holoviews does not like mixing Curve and Spread for the same stream
def roi_spreads(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]=  hv.Spread((wl,mean[param],std[param]),fill_alpha=0.3)

    return hv.NdOverlay(curves)

mean=hv.DynamicMap(roi_curves,streams=[box_stream])
std =hv.DynamicMap(roi_spreads, streams=[box_stream])    
hlines = hv.HoloMap({wl[i]: hv.VLine(wl[i]) for i in range(Nwl)},third_dim )


hv.output(widget_location='top_left')

# visualize and play
graphs = ((mean* std *hlines).relabel(param))
layout = (im * polys +graphs    ).opts(opts.Image(tools=['hover']),
    opts.Curve(width=750,height=500, framewise=True,xlim=(400,2500),tools=['hover']), 
    opts.Polygons(fill_alpha=0.2, color='green',line_color='black'), 
    opts.VLine(color='black')).cols(2)
layout 


# In[156]:


from aeronet_visu import data_loading as dl
dir = '/DATA/AERONET/OCv3/'
figdir= '/DATA/AERONET/fig'
aeronet_site = 'Bahia_Blanca'
file = aeronet_site+'_OCv3.lev15'

irr = dl.irradiance()
irr.load_F0()

param = 'Lwn'

# ---------------------------------------------
# Load data and convert into xarray
# ---------------------------------------------

df = dl.read(opj(dir, file)).read_aeronet_ocv3()

df = df.droplevel(0, 1)
# criteria to select scalar and spectrum values
criteria = df.columns.get_level_values(1) == ''
df_att = df.loc[:, criteria].droplevel(1, 1)
df_spec = df.loc[:, ~criteria]

ds = df_spec.stack().to_xarray()
ds = xr.merge([ds, df_att.to_xarray()])
del df

ds = ds.assign_coords({'level_1': ds.level_1.astype(float)}).rename({'level_1': "wl"})
ds['SZA'] = ds.Solar_Zenith_Angle.mean(axis=1)
ds['year']=ds['date.year']
ds['season']=ds['date.season']

wl = ds.wavelength * 1000
ds['Rrs'] = ds[param] / (irr.get_F0(wl) * 0.1)


# In[157]:


plt.figure(figsize=(10,5))

for i in range(3):
    for j in range(3):
        Rrs_l2.isel(x=25+i,y=25+j).plot(x='wl',color='black',alpha=0.5,lw=1.2)#,label='PRISMA')
ds.Rrs.sel(date='2021-08-14',method='nearest').plot(x='wl',marker='o',color='red',label='AERONET-OC')
plt.hlines(0,350,2500,ls=':',lw=0.5,color='black',zorder=0)
plt.ylabel('$R_{rs}\ (sr^{-1})$')
plt.legend()
plt.xlim(390,1100)


# ## Example of exploiation: compute NDWI for water pixel masking
# 

# In[41]:


# Compute NDWI
img=dc_l1c.Rtoa
green = img.sel(wl=565,method='nearest')
nir = img.sel(wl=865,method='nearest')
ndwi = (green - nir) / (green + nir)


# In[42]:


coarsening=1

# binary cmap
bcmap = mpl.colors.ListedColormap(['khaki', 'lightblue'])

def water_mask(ndwi, threshold=0):
    water = xr.where(ndwi > threshold, 1, 0)
    return water.where(~np.isnan(ndwi))

def plot_water_mask(ndwi,ax,threshold=0):
    water = water_mask(ndwi, threshold)
    #ax.set_extent(extent_val, proj)
    water.plot.imshow( cmap=bcmap,
                                  cbar_kwargs={'ticks': [0, 1], 'shrink': shrink})#extent=extent_val, transform=proj,
    ax.set_title(str(threshold)+' < NDWI')
    
fig = plt.figure(figsize=(20, 15))
fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.99,
                    hspace=0.05, wspace=0.05)
shrink = 0.8
    
ax = plt.subplot(2, 2, 1)#, projection=proj)
#ax.set_extent(extent_val, proj)
fig = ndwi[::coarsening, ::coarsening].plot.imshow(cmap=plt.cm.BrBG, robust=True,
                                   cbar_kwargs={'shrink': shrink})# extent=extent_val, transform=proj, 
# axes.coastlines(resolution='10m',linewidth=1)
ax.set_title('PRISMA, NDWI')

for i,threshold in enumerate([-0.2,0.,0.2]):
    ax = plt.subplot(2, 2, i+2)#, projection=proj)
    plot_water_mask(ndwi[::coarsening, ::coarsening],ax,threshold=threshold)

plt.show()



# ## Plot the top-of-atmosphere (TOA) radiance in mW/m2/nm

# In[43]:


threshold=-0.1

masked = Rrs.where(ndwi > threshold)


# In[45]:


masked
coarsening=1
gamma=0.2
brightness_factor = 1
fig = (masked[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])**gamma*brightness_factor).plot.imshow(rgb='wl',robust=True)#, subplot_kws=dict(projection= l1c.proj))
fig.axes.set(xticks=[], yticks=[])
fig.axes.set_ylabel('')
fig.axes.set_xlabel('')
fig
plt.show()


# In[47]:


fig = masked.isel(wl=[1,10,20,30,40,50,90,130,200]).plot.imshow(col='wl',col_wrap=3,robust=True,cmap=cc.cm.bky)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig


# In[49]:


from hgrs import metadata
auxdata = metadata()#wl=masked.wl)
wlref=2200
wl = img.wl
sunglint_eps = auxdata.sunglint_eps.interp(wl=wl)
rot = auxdata.rot.interp(wl=wl)


# In[50]:


sunglint_eps


# In[51]:


sunglintBRDF =  sunglint_eps['mean'] / sunglint_eps['mean'].sel(wl=slice(2150,2350)).mean(dim='wl')
sunglintBRDF.plot()


# In[52]:


aot550=0.1
ang_exp = 1.1

def aot_angstrom(wl, aotref, ang_exp,wlref=550):
    '''function for spectral variation of AOT'''
    
    return (wl/wlref)**-ang_exp * aotref

aot=aot_angstrom(wl,aot550,ang_exp)

fig, axs = plt.subplots()
aot.plot(ax=axs,label='Aerosol')
rot.plot(ls=':',label='Rayleigh',ax=axs)
axs.set_xlabel('Wavelength (nm)')
axs.set_ylabel('Optical thickness')
plt.legend()


# In[53]:


lut.sel(model='COPO_rh99').aot.plot(hue='aot_ref')
aot=lut.sel(model='COPO_rh99').aot.interp(aot_ref=0.4).interp(wl=img.wl/1000,method='cubic')


# In[54]:


def transmittance_dir(aot,sza=30,vza=10,rot=0):
    air_mass = 1/np.cos(np.radians(sza)) + 1/np.cos(np.radians(vza))
    return np.exp(-(rot+aot)*air_mass)
plt.figure()               
Tdir = transmittance_dir(aot,rot=rot)
Tdir.plot()


# In[81]:


sunglint_corr = Tdir * sunglintBRDF
Rcorr =masked - sunglint_corr * masked.sel(wl=slice(2150,2350)).mean(dim='wl')


# In[ ]:





# In[82]:


fig = Rcorr.isel(wl=[1,10,20,30,40,50,90,130,200]).plot.imshow(col='wl',col_wrap=3,vmin=0,robust=True,cmap=cc.cm.bky)
for ax in fig.axs.flat:
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel('')
    ax.set_xlabel('')
fig


# In[61]:


coarsening=1
brightness_factor = 7.5
(masked.Rtoa[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])*brightness_factor).plot.imshow(rgb='wl')#, subplot_kws=dict(projection= l1c.proj))


# In[62]:


#(Rcorr[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])*brightness_factor).plot.imshow(rgb='wl')#, subplot_kws=dict(projection= l1c.proj))
coarsening=1
plt.figure()
gamma=2
brightness_factor = 1
fig = (dc_l1c.Rtoa[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])**gamma*brightness_factor).plot.imshow(rgb='wl',robust=True)#, subplot_kws=dict(projection= l1c.proj))
(Rcorr[:, ::coarsening, ::coarsening].isel(wl=[30,20,10])).plot.imshow(ax=fig.axes,rgb='wl',robust=True)#, subplot_kws=dict(projection= l1c.proj))
fig.axes.set(xticks=[], yticks=[])
fig.axes.set_ylabel('')
fig.axes.set_xlabel('')
fig


# In[70]:


Rcorr.name='Rrs'
float(Rcorr.max())


# In[74]:


from holoviews import streams
import holoviews as hv
import panel as pn
import param
import numpy as np
import xarray as xr
hv.extension('bokeh')
from holoviews import opts

opts.defaults(
    opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    opts.Image(cmap='binary_r', width=800, height=700),
    opts.Labels(text_color='white', text_font_size='8pt', text_align='left', text_baseline='bottom'),
    opts.Path(color='white'),
    opts.Spread(width=900),
    opts.Overlay(show_legend=True))

raster=Rcorr
param='Rrs'
#raster = dc_l1c[param]  
cmax=0.01
third_dim = 'wl'
cmap = cc.cm.CET_L16

wl= raster.wl.data
Nwl = len(wl)
ds = hv.Dataset(raster.persist())
im= ds.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap= cmap,colorbar=True,clim=(0,cmax)).hist(bin_range=(0,cmax/2)) 

polys = hv.Polygons([])
box_stream = hv.streams.BoxEdit(source=polys)
dmap, dmap_std=[],[]

def roi_curves(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]= hv.Curve((wl,mean[param]),'Wavelength (nm)', param) 

    return hv.NdOverlay(curves)


# a bit dirty to have two similar function, but holoviews does not like mixing Curve and Spread for the same stream
def roi_spreads(data,ds=ds):    
    if not data or not any(len(d) for d in data.values()):
        return hv.NdOverlay({0: hv.Curve([],'Wavelength (nm)', param)})

    curves,envelope = {},{}
    data = zip(data['x0'], data['x1'], data['y0'], data['y1'])
    for i, (x0, x1, y0, y1) in enumerate(data):
        selection = ds.select(x=(x0, x1), y=(y0, y1))
        mean = selection.aggregate(third_dim, np.mean).data
        std = selection.aggregate(third_dim, np.std).data
        wl = mean.wl

        curves[i]=  hv.Spread((wl,mean[param],std[param]),fill_alpha=0.3)

    return hv.NdOverlay(curves)

mean=hv.DynamicMap(roi_curves,streams=[box_stream])
std =hv.DynamicMap(roi_spreads, streams=[box_stream])    
hlines = hv.HoloMap({wl[i]: hv.VLine(wl[i]) for i in range(Nwl)},third_dim )


hv.output(widget_location='top_left')

# visualize and play
graphs = ((mean* std *hlines).relabel(param))
layout = (im * polys +graphs    ).opts(
    opts.Curve(width=600, framewise=True,xlim=(400,2500),tools=['hover']), 
    opts.Polygons(fill_alpha=0.2, color='green',line_color='black'), 
    opts.VLine(color='black')).cols(2)
layout 


# In[ ]:




