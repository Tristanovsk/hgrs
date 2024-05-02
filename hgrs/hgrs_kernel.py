import os
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import xarray as xr

from numba import jit
from scipy import ndimage

import matplotlib.pyplot as plt

from multiprocessing import Pool  # Process pool
from multiprocessing import sharedctypes
import itertools
from scipy.optimize import least_squares

from . import auxdata

opj = os.path.join


class product():
    def __init__(self, l1c_obj=None, xcoarsen=20, ycoarsen=20):

        # spectral parameters
        self.wl_water_vapor = slice(800, 1300)
        self.wl_sunglint = slice(2150, 2250)
        self.wl_atmo = slice(950, 2450)
        self.wl_to_remove = [(935, 967), (1105, 1170), (1320, 1490), (1778, 2033), (2465, 2550)]
        self.wl_green = slice(540, 570)
        self.wl_nir = slice(850, 882)
        self.wl_1600 = slice(1580, 1650)
        self.wl_rgb = [30, 20, 10]

        # image chunking and coarsening parameters
        self.xcoarsen = xcoarsen
        self.ycoarsen = ycoarsen
        self.Npix_per_megapix = self.xcoarsen * self.ycoarsen
        self.block_size = 2
        # minimum percentage of water pixel within the mega-pixel to enable processing
        self.pixel_percentage = 20
        self.pixel_threshold = self.pixel_percentage / 100 * self.Npix_per_megapix

        # number of digits to keep for angle values
        self.ang_resol = 1

        # pre-computed auxiliary data
        self.dirdata = resource_filename(__package__, '../data/')
        self.abs_gas_file = opj(self.dirdata, 'lut', 'lut_abs_opt_thickness_normalized.nc')
        self.lut_file = opj(self.dirdata, 'lut', 'opac_osoaa_lut_v2.nc')
        self.water_vapor_transmittance_file = opj(self.dirdata, 'lut', 'water_vapor_transmittance.nc')

        # mask thresholding parameters
        self.sunglint_threshold = 0.11
        self.ndwi_threshold = 0.01
        self.green_swir_index_threshold = 0.1

        # atmosphere auxiliary data
        # TODO get them from CAMS
        self.pressure = 1010
        self.to3c = 6.5e-3
        self.tno2c = 3e-6
        self.tch4c = 1e-2
        self.psl = 1013
        self.coef_abs_scat = 0.3

        self.altitude = 0

        # xarray object to be processed
        self.raster = l1c_obj
        self.Rtoa = self.raster['Rtoa']
        self.RSR = self.raster.fwhm.to_dataframe()
        self.wl = self.Rtoa.wl
        self.sza_mean = np.nanmean(self.raster.sza)
        self.vza_mean = np.nanmean(self.raster.vza)
        self.raa_mean = np.nanmean(self.raster.raa)
        self.get_air_mass()

        self.Tg_other = None

        self.load_auxiliary_data()

    def load_auxiliary_data(self):

        # get LUT
        self.gas_lut = xr.open_dataset(self.abs_gas_file)
        self.aero_lut = xr.open_dataset(self.lut_file)
        # convert wavelength in nanometer
        self.aero_lut['wl'] = self.aero_lut['wl'] * 1000
        self.aero_lut['wl'].attrs['description'] = 'wavelength of simulation (nanometer)'
        self.Twv_lut = xr.open_dataset(self.water_vapor_transmittance_file)

        # get hgrs auxdata
        self.auxdata = auxdata(self.wl)

    def get_ndwi(self):
        green = self.Rtoa.sel(wl=self.wl_green).mean(dim='wl')
        nir = self.Rtoa.sel(wl=self.wl_nir).mean(dim='wl')
        self.ndwi = (green - nir) / (green + nir)

    def get_green_swir_index(self):
        green = self.Rtoa.sel(wl=self.wl_green).mean(dim='wl')
        b1600 = self.Rtoa.sel(wl=self.wl_1600).mean(dim='wl')
        self.green_swir_index = (green - b1600) / (green + b1600)

    def get_b2200(self):
        self.b2200 = self.Rtoa.sel(wl=self.wl_sunglint).mean(dim='wl')

    def apply_water_masks(self):
        self.get_ndwi()
        self.get_green_swir_index()
        self.get_b2200()
        self.raster['Rtoa_masked'] = self.Rtoa.where(self.ndwi > self.ndwi_threshold). \
            where(self.b2200 < self.sunglint_threshold). \
            where(self.green_swir_index > self.green_swir_index_threshold)

    def round_angles(self):
        for param in ['sza', 'vza', 'raa']:
            self.raster[param] = self.raster[param].round(self.ang_resol)

    def get_air_mass(self, raster_name='raster', round=True, digit_resol=3):
        raster = self.__dict__[raster_name]
        raster['air_mass'] = 1. / np.cos(np.radians(raster.sza)) + 1. / np.cos(np.radians(raster.vza))
        if round:
            raster['air_mass'] = raster['air_mass'].round(digit_resol)
        self.air_mass_mean = np.nanmean(raster['air_mass'].values)

    @staticmethod
    def remove_wl_dataarray(xarr, wl_to_remove, drop=True):
        xarr_ = xarr.isel(x=1, y=1)
        for wls in wl_to_remove:
            wl_min, wl_max = wls
            xarr_ = xarr_.where((xarr_.wl < wl_min) | (xarr_.wl > wl_max), drop=drop)
        wl_final = xarr_.wl.values
        return xarr.sel(wl=wl_final)

    @staticmethod
    def remove_wl_dataset(xds, wl_to_remove, variable='Rtoa', drop=True):
        xarr_ = xds[variable].isel(x=1, y=1)
        for wls in wl_to_remove:
            wl_min, wl_max = wls
            xarr_ = xarr_.where((xarr_.wl < wl_min) | (xarr_.wl > wl_max), drop=drop)
        wl_final = xarr_.wl.values
        return xds.sel(wl=wl_final)

    @staticmethod
    def Gamma2sigma(Gamma):
        '''Function to convert FWHM (Gamma) to standard deviation (sigma)'''
        return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)

    @staticmethod
    def gaussian(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def plot_rsr(self):

        wl_ref = np.linspace(360, 2550, 10000)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for mu, fwhm in self.RSR.iterrows():
            sig = self.Gamma2sigma(fwhm.values)
            rsr = self.gaussian(wl_ref, mu, sig)
            axs.plot(wl_ref, rsr, '-k', lw=0.5, alpha=0.4)
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Spectral response function')

        return fig

    def plot_angles(self, raster_name='raster',
                    figsize=(20, 4),
                    cmap=plt.cm.Spectral_r, **kwargs):
        raster = self.__dict__[raster_name]
        params = [raster.sza, raster.vza, raster.raa, raster.air_mass]
        titles = ['SZA', 'VZA', 'rel. AZI', 'Air mass']
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize)

        for i, ax in enumerate(axs):
            params[i].plot.imshow(ax=ax, robust=True, cmap=cmap, **kwargs)
            ax.set_title(titles[i])
            ax.set(xticks=[], yticks=[])
            ax.set_ylabel('')
            ax.set_xlabel('')
        return fig

    def plot_params(self, xds,
                    params=['aot_ref', 'aot_ref_std', 'brdfg', 'brdfg_std'],
                    shrink=0.8,
                    cmap=plt.cm.Spectral_r):

        ncols = len(params)
        fig_width = ncols * 5 + 2
        fig, axs = plt.subplots(1, ncols=ncols, figsize=(fig_width, 4))
        axs = axs.ravel()

        for i in range(4):
            xds[params[i]].plot.imshow(cmap=cmap, robust=True, vmin=0,  # vmax=0.201,
                                       cbar_kwargs={'shrink': shrink, 'label': params[i]},
                                       ax=axs[i])  # extent=extent_val, transform=proj,
            axs[i].set(xticks=[], yticks=[])
            axs[i].set_ylabel('')
            axs[i].set_xlabel('')
            axs[i].set_title(params[i])
        return fig

    def plot_masks(self, params=['cloud_mask', 'sunglint_mask', 'landcover_mask'],
                   vmax=12,
                   shrink=0.8,
                   cmap=plt.cm.Spectral_r):

        ncols = len(params)
        fig_width = ncols * 5 + 1
        fig, axs = plt.subplots(1, ncols=ncols, figsize=(fig_width, 4))

        fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.99,
                            hspace=0.15, wspace=0.15)
        axs = axs.ravel()

        for i, param in enumerate(params):
            self.raster[param].plot.imshow(cmap=cmap, vmax=vmax, robust=True,
                                           cbar_kwargs={'shrink': shrink},
                                           ax=axs[i])  # extent=extent_val, transform=proj,
            axs[i].set(xticks=[], yticks=[])
            axs[i].set_ylabel('')
            axs[i].set_xlabel('')
            axs[i].set_title(param)
        return fig

    def rgb(self, variable='Rtoa', raster_name='raster', gamma=0.5, brightness_factor=1, **kwargs):
        fig = (self.__dict__[raster_name][variable].isel(
            wl=self.wl_rgb) ** gamma * brightness_factor).plot.imshow(rgb='wl', robust=True, **kwargs)
        fig.axes.set(xticks=[], yticks=[])
        fig.axes.set_ylabel('')
        fig.axes.set_xlabel('')
        return fig

    def plot_water_pix_number(self, cmap=plt.cm.Spectral_r, **kwargs):
        try:
            fig = self.coarse_masked_raster['water_pixel_number'].plot.imshow(cmap=cmap, robust=True, **kwargs)
            fig.axes.set(xticks=[], yticks=[])
            fig.axes.set_ylabel('')
            fig.axes.set_xlabel('')
            return fig
        except:
            print('please apply algo.get_coarse_masked_raster() before')


class algo(product):

    def __init__(self, l1c_obj=None, xcoarsen=20, ycoarsen=20):
        product.__init__(self, l1c_obj, xcoarsen, ycoarsen)

    def get_pressure(self, alt, psl):
        '''Compute the pressure for a given altitude
           alt : altitude in meters (float or np.array)
           psl : pressure at sea level in hPa
           palt : pressure at the given altitude in hPa'''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    def get_coarse_raster(self, variables=['sza', 'vza', 'raa', 'air_mass', 'Rtoa']):
        self.coarse_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen).mean()

    def get_coarse_masked_raster(self, variables=['sza', 'vza', 'raa', 'air_mass', 'Rtoa_masked']):
        self.coarse_masked_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen).mean()
        self.coarse_masked_raster['water_pixel_number'] = self.raster['Rtoa_masked']. \
            isel(wl=slice(10, 20)).mean(dim='wl'). \
            coarsen(x=self.xcoarsen, y=self.ycoarsen).count()

    def get_gaseous_optical_thickness(self):
        gas_lut = self.gas_lut

        ot_o3 = gas_lut.o3 * self.to3c
        ot_ch4 = gas_lut.ch4 * self.tch4c
        ot_no2 = gas_lut.no2 * self.tno2c
        ot_air = (gas_lut.co + self.coef_abs_scat * gas_lut.co2 +
                  self.coef_abs_scat * gas_lut.o2 +
                  self.coef_abs_scat * gas_lut.o4) * self.pressure / 1000
        self.abs_gas_opt_thick = ot_ch4 + ot_no2 + ot_o3 + ot_air

    def get_gaseous_transmittance(self):

        self.get_gaseous_optical_thickness()
        wl_ref = self.gas_lut.wl  # .values
        Tg = np.exp(- self.air_mass_mean * self.abs_gas_opt_thick)
        prisma_rsr = self.raster.fwhm.to_dataframe()
        Tg_int = []
        for mu, fwhm in prisma_rsr.iterrows():
            sig = self.Gamma2sigma(fwhm.values)
            rsr = self.gaussian(wl_ref, mu, sig)
            Tg_ = (Tg * rsr).integrate('wl') / np.trapz(rsr, wl_ref)
            Tg_int.append(Tg_.values)

        self.Tg_other = xr.DataArray(Tg_int, name='Ttot', coords={'wl': self.raster.wl.values})

    def other_gas_correction(self, raster_name='coarse_masked_raster', variable='Rtoa_masked'):
        raster = self.__dict__[raster_name]
        attrs = raster[variable].attrs
        if attrs.__contains__('other_gas_correction'):
            if attrs['other_gas_correction']:
                print('raster ' + raster_name + '.' + variable + ' is already corrected for other gases transmittance')
                print('set attribute other_gas_correction to False to proceed anyway')
                return
        if self.Tg_other is None:
            self.get_gaseous_transmittance(self.air_mass_mean)
        raster[variable] = raster[variable] / self.Tg_other
        raster[variable].attrs['other_gas_correction'] = True

    def water_vapor_correction(self, raster_name='coarse_masked_raster', variable='Rtoa_masked'):
        raster = self.__dict__[raster_name]
        attrs = raster[variable].attrs
        if attrs.__contains__('water_vapor_correction'):
            if attrs['other_gas_correction']:
                print('raster ' + raster_name + '.' + variable + ' is already corrected for water vapor transmittance')
                print('set attribute other_gas_correction to False to proceed anyway')
                return

        if self.Twv_raster is None:
            print('xarray of water vapor transmittance is not set, please run get_wv_transmittance_raster(tcwv_raster)')
            return
        raster[variable] = raster[variable] / self.Twv_raster
        raster[variable].attrs['water_vapor_correction'] = True

    def get_wv_transmittance_raster(self, tcwv_raster):
        tcwv_vals = tcwv_raster.tcwv.round(1)
        tcwvs = np.unique(tcwv_vals)
        tcwvs = tcwvs[~np.isnan(tcwvs)]
        # TODO improve for air_mass raster
        Twvs = self.Twv_lut.Twv.interp(air_mass=self.air_mass_mean).interp(tcwv=tcwvs, method='linear').drop('air_mass')
        self.Twv_raster = Twvs.interp(tcwv=tcwv_vals, method='nearest')

    def get_full_resolution(self, xarr):
        return xarr.interp(x=self.raster.x, y=self.raster.y)


class solver():
    def __init__(self):
        pass

    def errFit(self, hess_inv, resVariance):
        '''
        Error/uncertainty of the estimated parameters
        :param resVariance:
        :return:
        '''
        return np.sqrt(np.diag(hess_inv * resVariance))

    def conv_mapping(self, x):
        """
        Nan-mean convolution
        """
        # get index of central pixel
        idx = len(x) // 2
        if np.isnan(x[idx]) and not np.isnan(np.delete(x, idx)).all():
            return np.nanmean(np.delete(x, idx))
        elif np.isnan(np.delete(x, idx)).all():
            return x[idx]
        else:
            return np.nanmean(x)

    @staticmethod
    @jit(nopython=True)
    def filter2d(image, weight, windows):
        '''
         Function to convolve parameter image with uncertainty image
        :param image: parameter image
        :param weight: uncertainty image
        :param windows: size of the window for convolution
        :return: convolved result with same shape as image

        '''
        M, N = np.shape(image)
        Mf, Nf = windows
        Mf2 = Mf // 2
        Nf2 = Nf // 2
        threshold = 0
        result = image
        for i in range(M):
            for j in range(N):
                num = 0.0
                norm = 0.0
                if weight[i, j] > threshold:
                    for ii in range(Mf):
                        ix = i - Mf2 + ii
                        if ix < M:
                            for jj in range(Nf):

                                iy = j - Nf2 + jj
                                if iy < N:
                                    wgt = weight[ix, iy]
                                    if wgt > 0.:
                                        num += (wgt * image[ix, iy])
                                        norm += wgt
                    result[i, j] = num / norm
        return result


class water_vapor(solver):

    def __init__(self, prod,
                 raster_name='coarse_masked_raster',
                 variable='Rtoa_masked'):

        self.prod = prod
        self.raster = prod.__dict__[raster_name]
        self.air_mass = prod.air_mass_mean
        # get data for the subset of "water vapor" wavelengths
        data = self.raster[variable].sel(wl=prod.wl_water_vapor)
        self.data = data
        self.width, self.height, self.nwl = data.shape
        self.x = data.x
        self.y = data.y
        self.wl = data.wl

        # TODO improve to process the air mass raster instead of scalar mean value
        self.Twv_ = prod.Twv_lut.Twv.sel(wl=self.wl).interp(air_mass=self.air_mass)
        self.Twv_['wl'] = self.Twv_['wl'] / 1000
        self.wl_mic = self.Twv_.wl.values

    def toa_simu(self, wl, Twv, tcwv, a, b):
        '''wl in micron
        '''
        # print(Twv.tcwv)
        return Twv.interp(tcwv=tcwv, method='linear').values * (a * wl + b)

    def toa_simu2(self, wl, Twv, tcwv, c0, c1, c2, c3):

        return c0 * np.exp(-c1 * wl ** -c2) * self.Twv_.interp(tcwv=tcwv).values \
            + c3 * self.wl_ ** -3 * self.Twv_.interp(tcwv=0.3 * tcwv).values

    def func(self, x, Twv, wl, y):
        return self.toa_simu(wl, Twv, *x) - y

    def func2(self, x, Twv, wl, y):
        return self.toa_simu2(wl, Twv, *x) - y

    def solve(self, x0=[2, -0.04, 0.1]):

        result = np.ctypeslib.as_ctypes(np.full((self.width, self.height, 6), np.nan))
        shared_array = sharedctypes.RawArray(result._type_, result)
        self.x0 = x0
        data = self.data
        height = self.height
        width = self.width
        block_size = self.prod.block_size
        pixel_threshold = self.prod.pixel_threshold
        if list(self.raster.keys()).__contains__('water_pixel_number'):
            water_pixel_number = self.raster.water_pixel_number
        else:
            water_pixel_number = None

        global chunk_process

        def chunk_process(args):

            window_x, window_y = args

            tmp = np.ctypeslib.as_array(shared_array)
            # x0 = [20, -0.04, 0.1]
            for ix in range(window_x, min(width, window_x + block_size)):
                for iy in range(window_y, min(height, window_y + block_size)):
                    if water_pixel_number is not None:
                        if water_pixel_number.isel(x=ix, y=iy).values < pixel_threshold:
                            continue

                    y = data.isel(x=ix, y=iy).dropna(dim='wl')
                    # sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')
                    # TODO put solver parameter in self instance
                    res_lsq = least_squares(self.func, self.x0, args=(self.Twv_, self.wl_mic, y),
                                            bounds=([0, -10, 0], [60, 1, 1]),
                                            diff_step=1e-2, xtol=1e-2, ftol=1e-2, max_nfev=20)
                    xres = res_lsq.x
                    resVariance = (res_lsq.fun ** 2).sum() / (len(res_lsq.fun) - len(res_lsq.x))
                    hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
                    try:
                        hess_inv = np.linalg.inv(hess)
                        std = self.errFit(hess_inv, resVariance)
                    except:
                        std = [np.nan, np.nan, np.nan]
                    tmp[ix, iy, :] = [*xres, *std]
            return

        window_idxs = [(i, j) for i, j in
                       itertools.product(range(0, width, block_size),
                                         range(0, height, block_size))]

        p = Pool()
        res = p.map(chunk_process, window_idxs)
        result = np.ctypeslib.as_array(shared_array)

        self.water_vapor = xr.Dataset(dict(tcwv=(["y", "x"], result[:, :, 0].T),
                                           tcwv_std=(["y", "x"], result[:, :, 3].T)),
                                      coords=dict(
                                          x=self.x,
                                          y=self.y),
                                      attrs=dict(
                                          description="Fitted Total Columnar Water vapor; warning for transmittance computation only",
                                          units="kg/m**2")
                                      )


class aerosol(solver):

    def __init__(self, prod,
                 aerosol_model='COAV_rh70',
                 first_guess=[0.01,0],
                 aot550_limits=[0.002,1.2],
                 raster_name='coarse_masked_raster',
                 variable='Rtoa_masked'):

        self.prod = prod
        self.aerosol_model = aerosol_model
        self.raster = prod.__dict__[raster_name]
        self.auxdata = prod.auxdata
        self.aero_lut = prod.aero_lut

        # set box limits in aod550 for non-linear optimization
        self.aod550_min = aot550_limits[0]
        self.aod550_max = aot550_limits[1]
        self.first_guess = first_guess

        # get full resolution parameters
        self.xfull = prod.raster.x
        self.yfull = prod.raster.y

        # get data for the subset of "black water" wavelengths
        data = self.raster[variable].sel(wl=prod.wl_atmo)
        self.data = data
        self.width, self.height, self.nwl = data.shape
        self.x = data.x
        self.y = data.y
        self.wl = data.wl

        self.sza = prod.sza_mean
        self.vza = prod.vza_mean
        self.raa = prod.raa_mean
        self.raa_lut = (180 - self.raa) % 360
        self.air_mass = prod.air_mass_mean
        self.pressure = prod.pressure

        # process parameters
        self.block_size = self.prod.block_size
        self.pixel_threshold = self.prod.pixel_threshold
        self.wl_sunglint = self.prod.wl_sunglint

        self.prepare_lut(self.wl)

    def prepare_lut(self, wl):

        auxdata = self.auxdata
        sza = self.sza
        vza = self.vza
        raa_lut = self.raa_lut

        self.sunglint_eps = auxdata.sunglint_eps.interp(wl=wl)
        self.rot = auxdata.rot.interp(wl=wl) * self.pressure / self.auxdata.pressure_rot_ref

        aot_refs = np.logspace(-3, np.log10(1.5), 100)
        self.aot_lut = self.aero_lut.sel(model=self.aerosol_model).aot.interp(wl=wl, method='quadratic').interp(
            aot_ref=aot_refs,
            method='quadratic').dropna('aot_ref')  # sel(wl=wl_glint)

        norm_radiance = self.aero_lut.sel(model=self.aerosol_model
                                          ).I.interp(vza=vza, azi=raa_lut, method='linear'
                                                     ).interp(sza=sza, method='quadratic').squeeze()
        self.Rtoa_lut = norm_radiance.interp(wl=wl, method='quadratic') \
                            .interp(aot_ref=aot_refs, method='quadratic').dropna('aot_ref') / np.cos(np.radians(sza))

    def transmittance_dir(self, aot, M, rot=0):
        return np.exp(-(rot + aot) * M)

    def toa_simu(self, aot, rot, Rtoa_lut, sunglint_eps, aot_ref, BRDFg):
        '''
        '''
        aot = aot.interp(aot_ref=aot_ref)
        Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
        Tdir = self.transmittance_dir(aot, self.air_mass, rot=rot)
        sunglint_corr = Tdir * sunglint_eps
        Rdir = sunglint_corr * BRDFg / Tdir.sel(wl=self.wl_sunglint).mean(dim='wl')
        # sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])

        return Rdiff + Rdir

    def func(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        return (self.toa_simu(aot, rot, Rtoa_lut, sunglint_eps, *x) - y)  # /sigma

    def solve(self, x0=[0.005, 0.]):

        result = np.ctypeslib.as_ctypes(np.full((self.width, self.height, 4), np.nan))
        shared_array = sharedctypes.RawArray(result._type_, result)
        # TODO clean up method to assign first guess
        self.x0 = x0
        self.x0 = self.first_guess

        data = self.data
        height = self.height
        width = self.width

        if list(self.raster.keys()).__contains__('water_pixel_number'):
            water_pixel_number = self.raster.water_pixel_number
        else:
            water_pixel_number = None

        global chunk_process

        def chunk_process(args):
            window_x, window_y = args
            tmp = np.ctypeslib.as_array(shared_array)

            for ix in range(window_x, min(width, window_x + self.block_size)):
                for iy in range(window_y, min(height, window_y + self.block_size)):
                    if water_pixel_number is not None:
                        if water_pixel_number.isel(x=ix, y=iy).values < self.pixel_threshold:
                            continue
                    x0 = self.x0
                    y = data.isel(x=ix, y=iy).dropna(dim='wl')
                    # sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')

                    res_lsq = least_squares(self.func, x0,
                                            args=(self.aot_lut, self.rot, self.Rtoa_lut, self.sunglint_eps, y),
                                            bounds=([self.aod550_min, 0], [self.aod550_max, 1.3]), diff_step=1e-3, xtol=1e-3, ftol=1e-3,
                                            max_nfev=20)
                    # except:
                    # print(wl_,aot_,rot_,Rtoa_lut_,sunglint_eps_, y)
                    #    break
                    xres = res_lsq.x
                    resVariance = (res_lsq.fun ** 2).sum() / (len(res_lsq.fun) - len(res_lsq.x))
                    hess = np.matmul(res_lsq.jac.T, res_lsq.jac)

                    try:
                        hess_inv = np.linalg.inv(hess)
                        std = self.errFit(hess_inv, resVariance)
                    except:
                        std = [np.nan, np.nan]
                    tmp[ix, iy, :] = [*xres, *std]

        window_idxs = [(i, j) for i, j in
                       itertools.product(range(0, self.width, self.block_size),
                                         range(0, self.height, self.block_size))]

        p = Pool()
        res = p.map(chunk_process, window_idxs)
        result = np.ctypeslib.as_array(shared_array)

        self.aero_img = xr.Dataset(dict(aot_ref=(["y", "x"], result[:, :, 0].T),
                                        brdfg=(["y", "x"], result[:, :, 1].T),
                                        aot_ref_std=(["y", "x"], result[:, :, 2].T),
                                        brdfg_std=(["y", "x"], result[:, :, 3].T)
                                        ),
                                   coords=dict(
                                       x=self.x,
                                       y=self.y),
                                   attrs=dict(
                                       description="aerosol and sunglint retrieval from coarse resolution data",
                                       aerosol_model=self.aerosol_model)
                                   )

    def smoothing(self, windows=np.array([15, 15]), mask=np.ones((3, 3))):

        weights = (1 / self.aero_img['aot_ref_std'] ** 2).values
        param = self.aero_img['aot_ref'].values
        aot_ref_smoothed = self.filter2d(param, weights, windows)
        res = ndimage.generic_filter(aot_ref_smoothed, function=self.conv_mapping, footprint=mask, mode='nearest')

        self.aero_img['aot_ref_smoothed'] = xr.DataArray(res, coords=dict(y=self.aero_img.y, x=self.aero_img.x))

    def get_aot_full_resolution(self):
        self.aot_ref_full = \
            self.aero_img['aot_ref_smoothed'].interp(x=self.xfull, y=self.yfull,
                                                     method='linear', kwargs={"fill_value": "extrapolate"})

    def get_atmo_parameters(self, wl):
        # get LUT for desired wavelengths
        self.prepare_lut(wl)
        self.smoothing()
        self.get_aot_full_resolution()

        # construct aot raster
        aot_ref_vals = self.aot_ref_full.round(3)
        aot_refs = np.unique(aot_ref_vals)
        aot_refs = aot_refs[~np.isnan(aot_refs)]
        # if rounded aot_ref has unique value
        if len(aot_refs) == 1:
            aot_refs = np.concatenate([ aot_refs, 1.2 * aot_refs])
        aots = self.aot_lut.interp(aot_ref=aot_refs, method='linear')
        aots = aots.interp(aot_ref=aot_ref_vals, method='nearest')

        aots.name = 'aot'
        aots.attrs['description'] = 'spectral aerosol optical thickness'

        # construct raster for diffuse atmospheric reflectance
        Rdiffs = self.Rtoa_lut.interp(aot_ref=aot_refs, method='linear')
        Rdiffs = Rdiffs.interp(aot_ref=aot_ref_vals, method='nearest')
        Rdiffs.name = 'Rtoa_diff'
        Rdiffs.attrs['description'] = 'top-of-atmosphere atmosphere reflectance'

        # construct raster for direct transmittance due to rayleigh and aerosol
        Tdirs = self.transmittance_dir(aots, self.air_mass, rot=self.rot)
        Tdirs.name = 'Tdir'
        Tdirs.attrs['description'] = 'direct transmittance due to rayleigh and aerosol for total air mass'

        # merge into dataset
        self.atmo_img = xr.merge([aots, Rdiffs, Tdirs])
        self.atmo_img.attrs['description'] = "atmospheric parameters for rayleigh and aerosol components",
        self.atmo_img.attrs['aerosol_model'] = self.aerosol_model
