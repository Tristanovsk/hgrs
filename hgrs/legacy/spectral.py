"""Legacy spectral response implementation retained for API compatibility.

This module preserves the spectral response formulas and behavior from
hGRS 1.1.2. Keep numerical changes out of structural moves from this module.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(fastmath=True)
def Gamma2sigma(Gamma):
    '''Function to convert FWHM (Gamma) to standard deviation (sigma)'''
    return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)


@njit(fastmath=True)
def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


@njit(fastmath=True)
def super_gaussian(x,
                   amplitude=1.0,
                   mu=0.0,
                   sigma=1.0,
                   expon=10.0):
    '''
    Super-Gaussian distribution:
    super_gaussian(x, amplitude, mu, sigma, expon) =
        (amplitude/(sqrt(2*pi)*sigma)) * exp(-abs(x-mu)**expon / (2*sigma**expon))
    :param x:
    :param amplitude:
    :param mu:
    :param sigma:
    :param expon:
    :return:
    '''

    sigma = max(1.e-15, sigma)
    return amplitude / (np.sqrt(2 * np.pi) * sigma) * \
        np.exp(-np.abs(x - mu) ** expon / (2 * sigma ** expon))


@njit(fastmath=True)
def super_gaussian_fwhm2sigma(fwhm,
                              expon):
    '''
    Function to convert FWHM to standard deviation (sigma) of the super-gaussian distribution
    :param fwhm:
    :param expon:
    :return:
    '''
    return fwhm / 2 * (2 * np.log(2)) ** (-1 / expon)


class Spectral():
    def __init__(self,
                 central_wl,
                 fwhm):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param central_wl: numpy array of the central wavelengths
        :param fwhm: scalar or numpy array containing full width at half maximum in nm                :param info: optional parameter to feed the attributes of the output xarray
        :return:
        '''
        self.central_wl = central_wl
        if not isinstance(fwhm, np.ndarray):
            fwhm = np.array([fwhm] * len(central_wl))
        fwhm = xr.DataArray(fwhm, name='fwhm',
                            coords={'wl': central_wl},
                            attrs={
                                'definition': 'full width at half maximum of spectral responses modeled as gaussian distributions'})
        self.fwhm = fwhm

    def plot_rsr(self):

        wl_ref = np.linspace(360, 2550, 10000)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        for mu, fwhm in self.fwhm.groupby('wl'):
            sig = self.Gamma2sigma(fwhm.values)
            rsr = self.gaussian(wl_ref, mu, sig)
            axs.plot(wl_ref, rsr, '-k', lw=0.5, alpha=0.4)
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Spectral response function')

        return fig

    @staticmethod
    @njit(parallel=True)
    def convolve_(
            wl_signal,
            signal,
            wl,
            fwhm,
    ):
        '''
        Convolution assuming Dirac for signal source spectral response
        :paral wl_signal: wavelength array of spectral signal
        :param signal: numpy of signal to convolve, coord=wl_signal
        :param wl: numpy of wavelength coordinates of signal
        :param fwhm: numpy with data=fwhm containing full width at half maximum in nm
        :return: numpy of convoluted signal
        '''
        Nwl = len(wl)
        signal_ = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = Gamma2sigma(fwhm[ii])
            rsr = gaussian(wl_signal, wl[ii], sig)
            signal_[ii] = np.trapezoid((signal * rsr), wl_signal) / np.trapezoid(rsr, wl_signal)
        return signal_

    @staticmethod
    @njit(parallel=True)
    def convolve2_(
            wl_signal,
            signal,
            wl,
            fwhm,
            expon=2.,
            threshold=1e-6
    ):
        '''
        Convolution assuming Dirac for signal source spectral response
        :paral wl_signal: wavelength array of spectral signal
        :param signal: numpy of signal to convolve, coord=wl_signal
        :param wl: numpy of wavelength coordinates of signal
        :param fwhm: numpy with data=fwhm containing full width at half maximum in nm
        :param threshold: minimum values of the response function to be included in the convolution
        :return: numpy of convoluted signal
        '''

        Nwl = len(wl)
        response = np.full((Nwl), np.nan, dtype=np.float32)
        for ii in prange(len(fwhm)):
            sig = super_gaussian_fwhm2sigma(fwhm[ii], expon)
            rsr = super_gaussian(wl_signal, mu=wl[ii], sigma=sig, expon=expon)

            # remove values above a given threshold to speed up computation
            idx = rsr > threshold
            wl_signal_ = wl_signal[idx]
            signal_ = signal[idx]
            rsr = rsr[idx]

            response[ii] = np.trapezoid((signal_ * rsr), wl_signal_) / np.trapezoid(rsr, wl_signal_)
        return response

    def convolve2(self,
                  signal,
                  name='signal',
                  expon=3,
                  threshold=1e-4,
                  info={}):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param signal: xarray spectral signal to convolve, coord=wl
        :param fwhm: xarray with data=fwhm containing full width at half maximum in nm, and coords=wl
        :param info: optional parameter to feed the attributes of the output xarray
        :param threshold: minimum values of the response function to be included in the convolution
        :return:
        '''

        wl_ref = signal.wl.values
        fwhm = self.fwhm.values
        wl = self.fwhm.wl.values
        xdims = signal.dims
        attrs = signal.attrs
        name = signal.name
        if len(xdims) == 1:
            signal_int = self.convolve2_(wl_ref, signal.values, wl, fwhm, expon, threshold=threshold)
            signal_int = xr.DataArray(signal_int, name=name,
                                      coords={'wl': self.fwhm.wl.values},
                                      attrs=attrs)

        else:
            # to handle multidimensional xarray
            xdims = np.array(xdims)
            xdims = xdims[xdims != 'wl']

            xsignal_int = []
            for dim in xdims:
                xsignal_int_ = []
                for value, signal_ in signal.groupby(dim):
                    # print(dim, value)
                    signal_ = signal_.squeeze()
                    _ = self.convolve2_(signal_.wl.values, signal_.values, wl, fwhm, expon)
                    _ = xr.Dataset({name: (['wl'], _)},
                                   coords={'wl': wl,
                                           dim: value})
                    xsignal_int_.append(_)
                xsignal_int.append(xr.concat(xsignal_int_, dim=dim))
            signal_int = xr.merge(xsignal_int)  # .to_dataarray()
            signal_int.attrs = attrs

        return signal_int

    def convolve(self,
                 signal,
                 name='signal',
                 info={}):
        '''
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param signal: xarray spectral signal to convolve, coord=wl
        :param fwhm: xarray with data=fwhm containing full width at half maximum in nm, and coords=wl
        :param info: optional parameter to feed the attributes of the output xarray
        :return:
        '''

        wl_ref = signal.wl.values
        fwhm = self.fwhm.values
        wl = self.fwhm.wl.values
        xdims = signal.dims

        if len(xdims) == 1:
            signal_int = self.convolve_(wl_ref, signal.values, wl, fwhm)
            signal_int = xr.DataArray(signal_int, name=name,
                                      coords={'wl': self.fwhm.wl.values},
                                      attrs=info)

        else:
            # to handle multidimensional xarray
            xdims = np.array(xdims)
            xdims = xdims[xdims != 'wl']

            xsignal_int = []
            for dim in xdims:
                xsignal_int_ = []
                for value, signal_ in signal.groupby(dim):
                    # print(dim, value)
                    signal_ = signal_.squeeze()
                    _ = self.convolve_(signal_.wl.values, signal_.values, wl, fwhm)
                    _ = xr.Dataset({name: (['wl'], _)},
                                   coords={'wl': wl,
                                           dim: value})
                    xsignal_int_.append(_)
                xsignal_int.append(xr.concat(xsignal_int_, dim=dim))
            signal_int = xr.merge(xsignal_int).to_dataarray()
            signal_int.attrs = info

        return signal_int
