import os
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import xarray as xr

from multiprocessing import Pool  # Process pool
from multiprocessing import sharedctypes
import itertools
from scipy.optimize import least_squares

opj = os.path.join


class product():
    def __init__(self, l1c_obj=None):
        # spectral parameters
        self.wl_water_vapor = slice(800, 1300)
        self.wl_sunglint = slice(2150, 2250)
        self.wl_atmo = slice(950, 2250)
        self.wl_to_remove = [(935, 967), (1105, 1170), (1320, 1490), (1778, 2033)]
        self.wl_green = slice(540, 570)
        self.wl_nir = slice(850, 882)
        self.wl_1600 = slice(1580, 1650)
        self.wl_rgb = [30, 20, 10]

        # image chunking and coarsening parameters
        self.xcoarsen = 20
        self.ycoarsen = 20
        self.block_size = 2
        # number of digits to keep for angle values
        self.ang_resol = 1

        # pre-computed auxiliary data
        self.dirdata = resource_filename(__package__, '../data/')
        self.abs_gas_file = opj(self.dirdata, 'lut', 'lut_abs_opt_thickness_normalized.nc')
        self.lut_file = opj(self.dirdata, 'lut', 'opac_osoaa_lut_v2.nc')
        self.pressure_rot_ref = 1013.25

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

        self.Tg_other = None

    def load_metadata(self):
        self.gas_lut = xr.open_dataset(self.abs_gas_file)
        self.aero_lut = xr.open_dataset(self.lut_file)

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

    def get_air_mass(self, round=True, digit_resol=3):
        self.air_mass = 1. / np.cos(np.radians(self.raster.sza)) + 1. / np.cos(np.radians(self.raster.vza))
        if round:
            self.air_mass = self.air_mass.round(digit_resol)
        self.air_mass_mean = np.nanmean(self.air_mass)

    @staticmethod
    def remove_wl(xarr, wl_to_remove):
        for wls in wl_to_remove:
            wl_min, wl_max = wls
            xarr = xarr.where((xarr.wl < wl_min) | (xarr.wl > wl_max))
        xarr = xarr.where((xarr.wl < 2450))
        return xarr

    @staticmethod
    def Gamma2sigma(Gamma):
        '''Function to convert FWHM (Gamma) to standard deviation (sigma)'''
        return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)

    @staticmethod
    def gaussian(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def rgb(self, variable='Rtoa', raster_name='raster', gamma=0.5, brightness_factor=1, **kwargs):
        fig = (self.__dict__[raster_name][variable].isel(
            wl=self.wl_rgb) ** gamma * brightness_factor).plot.imshow(rgb='wl', robust=True, **kwargs)
        fig.axes.set(xticks=[], yticks=[])
        fig.axes.set_ylabel('')
        fig.axes.set_xlabel('')
        return fig


#
#
# class gas_corr(product):
#     # Gaseous transmittance
#     wl_ref = gas_lut.wl  # .values
#     ot_o3 = gas_lut.o3 * to3c
#     ot_ch4 = gas_lut.ch4 * tch4c
#     ot_no2 = gas_lut.no2 * tno2c
#     ot_air = (
#                          gas_lut.co + coef_abs_scat * gas_lut.co2 + coef_abs_scat * gas_lut.o2 + coef_abs_scat * gas_lut.o4) * pressure / 1000
#     ot_other = ot_ch4 + ot_no2 + ot_o3 + ot_air
#
#     Tg = np.exp(-M * ot_other)
#
#     Tg_int = []
#     for mu, fwhm in prisma_rsr.iterrows():
#         sig = Gamma2sigma(fwhm.values)
#         rsr = gaussian(wl_ref, mu, sig)
#         Tg_ = (Tg * rsr).integrate('wl') / np.trapz(rsr, wl_ref)
#         Tg_int.append(Tg_.values)
#
#     Tg_other = xr.DataArray(Tg_int, name='Ttot', coords={'wl': dc_l1c.wl.values})
#     def toa_simu(wl, Twv, tcwv, a, b):
#         '''wl in micron
#         '''
#         # print(Twv.tcwv)
#         return Twv.interp(tcwv=tcwv, method='linear').values * (a * wl + b)
#
#
#     def toa_simu2(wl, Twv, tcwv, c0, c1, c2, c3):
#         return c0 * np.exp(-c1 * wl ** -c2) * Twv_.interp(tcwv=tcwv).values + c3 * wl_ ** -3 * Twv_.interp(
#             tcwv=0.3 * tcwv).values
#
#
#     def fun(x, Twv, wl, y):
#         return toa_simu(wl, Twv, *x) - y
#
#
#     def fun2(x, Twv, wl, y):
#         return toa_simu2(wl, Twv, *x) - y
#

class algo(product):

    def __init__(self, l1c_obj=None):
        product.__init__(self, l1c_obj)

    def get_pressure(self, alt, psl):
        '''Compute the pressure for a given altitude
           alt : altitude in meters (float or np.array)
           psl : pressure at sea level in hPa
           palt : pressure at the given altitude in hPa'''

        palt = psl * (1. - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    def get_coarse_raster(self, variables=['sza', 'vza', 'raa', 'Rtoa']):
        self.coarse_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen).mean()

    def get_coarse_masked_raster(self, variables=['sza', 'vza', 'raa', 'Rtoa_masked']):
        self.coarse_masked_raster = self.raster[variables].coarsen(x=self.xcoarsen, y=self.ycoarsen).mean()
        self.coarse_masked_raster['pixel_number'] = self.raster['Rtoa_masked']. \
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
        attrs = self.__dict__[raster_name][variable].attrs
        if attrs.__contains__('other_gas_correction'):
            if attrs['other_gas_correction']:
                print('raster ' + raster_name + '.' + variable + ' is already corrected for other gases transmittance')
                print('set attribute other_gas_correction to False to proceed anyway')
                return
        if self.Tg_other is None:
            self.get_gaseous_transmittance()
        self.__dict__[raster_name][variable] = self.__dict__[raster_name][variable] / self.Tg_other
        self.__dict__[raster_name][variable].attrs['other_gas_correction'] = True


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


class water_vapor(solver):

    def __init__(self, prod, Twv_hires, raster_name='coarse_masked_raster',
                 variable='Rtoa_masked'):  # data,Twv_hires,pixel_number=None,pix_thresh=1,block_size=2):

        self.prod = prod
        # get data for the subset of "water vapor" wavelengths
        data = prod.__dict__[raster_name][variable].sel(wl=prod.wl_water_vapor)
        self.data = data
        self.width, self.height, self.nwl = data.shape
        self.x = data.x
        self.y = data.y
        self.wl = data.wl

        self.Twv_ = Twv_hires.sel(wl=self.wl)
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

    def solve(self, x0=[20, -0.04, 0.1]):

        result = np.ctypeslib.as_ctypes(np.full((self.width, self.height, 6), np.nan))
        shared_array = sharedctypes.RawArray(result._type_, result)

        height = self.height
        width = self.width
        block_size = self.prod.block_size
        data = self.prod.data
        pix_thresh = self.prod.pix_thresh
        pixel_number = self.prod.pixel_number

        global chunk_process

        def chunk_process(args):

            window_x, window_y = args

            tmp = np.ctypeslib.as_array(shared_array)
            x0 = [20, -0.04, 0.1]
            for ix in range(window_x, min(width, window_x + block_size)):

                for iy in range(window_y, min(height, window_y + block_size)):
                    if pixel_number is not None:
                        if pixel_number.isel(x=ix, y=iy).values < pix_thresh:
                            continue

                    y = data.isel(x=ix, y=iy).dropna(dim='wl')
                    # sigma = Rtoa_std.isel(x=ix,y=iy).dropna(dim='wl')
                    # TODO put solver parameter in self instance
                    res_lsq = least_squares(self.func, x0, args=(self.Twv_, self.wl_mic, y),
                                            bounds=([0, -10, 0], [60, 1, 1]),
                                            diff_step=1e-2, xtol=1e-2, ftol=1e-2, max_nfev=20)
                    x0 = res_lsq.x
                    resVariance = (res_lsq.fun ** 2).sum() / (len(res_lsq.fun) - len(res_lsq.x))
                    hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
                    try:
                        hess_inv = np.linalg.inv(hess)
                        std = self.errFit(hess_inv, resVariance)
                    except:
                        std = [np.nan, np.nan, np.nan]
                    tmp[ix, iy, :] = [*x0, *std]
            return

        window_idxs = [(i, j) for i, j in
                       itertools.product(range(0, self.width, self.block_size),
                                         range(0, self.height, self.block_size))]

        p = Pool()
        res = p.map(chunk_process, window_idxs)
        result = np.ctypeslib.as_array(shared_array)

        self.gas_img = xr.Dataset(dict(tcwv=(["y", "x"], result[:, :, 0].T),
                                       tcwv_std=(["y", "x"], result[:, :, 3].T)),
                                  coords=dict(
                                      x=self.x,
                                      y=self.y),

                                  attrs=dict(
                                      description="Fitted Total Columnar Water vapor; warning for transmittance computation only",
                                      units="kg/m**2")
                                  )


class aerosol(solver):

    def __init__(self, prod, raster_name='coarse_masked_raster', variable='Rtoa_masked'):
        self.prod = prod
        data = prod.__dict__[raster_name][variable]
        # remove undesired wavelength
