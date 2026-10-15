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
    r'''Function to convert FWHM (Gamma) to standard deviation (sigma)

    .. math::

       \sigma = \frac{\sqrt{2}\, \Gamma}{2 \sqrt{2 \ln 2}} = \frac{\Gamma}{2\sqrt{\ln 2}}

    .. note::

       This is :math:`\sqrt{2}` times the standard deviation of a Gaussian of full width at half maximum
       :math:`\Gamma`, :math:`\Gamma / (2\sqrt{2\ln 2})` (see :py:func:`super_gaussian_fwhm2sigma`
       with ``expon=2``).

    Examples
    --------
    >>> from hgrs.hgrs_kernel import Gamma2sigma
    >>> round(Gamma2sigma(10.), 4)
    6.0056
    '''
    return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)


@njit(fastmath=True)
def gaussian(x, mu, sigma):
    r'''Normalized Gaussian distribution

    .. math::

       g(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

    Examples
    --------
    >>> from hgrs.hgrs_kernel import gaussian
    >>> round(gaussian(0., 0., 1.), 4)
    0.3989
    '''
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


@njit(fastmath=True)
def super_gaussian(x,
                   amplitude=1.0,
                   mu=0.0,
                   sigma=1.0,
                   expon=10.0):
    r'''
    Super-Gaussian distribution

    .. math::

       g(x) = \frac{A}{\sigma\sqrt{2\pi}} \exp\left(-\frac{|x - \mu|^p}{2\sigma^p}\right)

    :param x: abscissa (e.g., wavelength)
    :param amplitude: :math:`A`
    :param mu: center :math:`\mu`
    :param sigma: width :math:`\sigma`
    :param expon: exponent :math:`p` (:math:`p = 2` for a Gaussian)
    :return: :math:`g(x)`

    Examples
    --------
    At a distance FWHM/2 of the center, the distribution is half of its maximum:

    >>> from hgrs.hgrs_kernel import super_gaussian, super_gaussian_fwhm2sigma
    >>> sigma = super_gaussian_fwhm2sigma(10., 6.)
    >>> round(super_gaussian(5., 1., 0., sigma, 6.) / super_gaussian(0., 1., 0., sigma, 6.), 4)
    0.5
    '''

    sigma = max(1.e-15, sigma)
    return amplitude / (np.sqrt(2 * np.pi) * sigma) * \
        np.exp(-np.abs(x - mu) ** expon / (2 * sigma ** expon))


@njit(fastmath=True)
def super_gaussian_fwhm2sigma(fwhm,
                              expon):
    r'''
    Function to convert FWHM to standard deviation (sigma) of the super-gaussian distribution

    .. math::

       \sigma = \frac{\Gamma}{2} \left(2 \ln 2\right)^{-1/p}

    :param fwhm: full width at half maximum :math:`\Gamma`
    :param expon: exponent :math:`p` of the super-Gaussian
    :return: :math:`\sigma`

    Examples
    --------
    >>> from hgrs.hgrs_kernel import super_gaussian_fwhm2sigma
    >>> round(super_gaussian_fwhm2sigma(10., 2.), 4)   # Gaussian: FWHM / (2 sqrt(2 ln 2))
    4.2466
    >>> round(super_gaussian_fwhm2sigma(10., 6.), 4)
    4.7351
    '''
    return fwhm / 2 * (2 * np.log(2)) ** (-1 / expon)


class Spectral():
    r'''
    Convolution of spectral signals with the spectral response functions (SRF) of the sensor bands

    .. math::

       X_i = \frac{\int X(\lambda)\, S_i(\lambda)\, d\lambda}{\int S_i(\lambda)\, d\lambda}

    where :math:`S_i` is a Gaussian (:py:meth:`convolve`) or a super-Gaussian (:py:meth:`convolve2`)
    centered on :math:`\lambda_i` with the full width at half maximum of the band.

    :param central_wl: numpy array of the central wavelengths
    :param fwhm: scalar or numpy array containing full width at half maximum in nm

    Examples
    --------
    A signal varying linearly with wavelength is not changed by symmetric response functions:

    >>> import numpy as np
    >>> import xarray as xr
    >>> from hgrs.hgrs_kernel import Spectral
    >>> wl = np.arange(400, 801, 1.)
    >>> signal = xr.DataArray((wl - 400) / 400, coords={'wl': wl})
    >>> spectral = Spectral(np.array([500., 600., 700.]), fwhm=10.)
    >>> spectral.convolve(signal).values.round(4)
    array([0.25, 0.5 , 0.75], dtype=float32)
    >>> spectral.convolve2(signal, expon=6).values.round(4)
    array([0.25, 0.5 , 0.75], dtype=float32)
    '''
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
        Convolve with super-Gaussian spectral responses of the bands (full width at half maximum ``fwhm``)

        :param signal: xarray spectral signal to convolve, coord=wl (other dimensions are looped over)
        :param name: kept for compatibility, the name of ``signal`` is used
        :param expon: exponent of the super-Gaussian (2 for a Gaussian)
        :param threshold: minimum values of the response function to be included in the convolution
        :param info: not used, the attributes of ``signal`` are kept
        :return: convolved signal on the central wavelengths of the bands (DataArray, or Dataset if
            ``signal`` has dimensions other than wl)
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
        Convolve with Gaussian spectral responses of the bands (full width at half maximum ``fwhm``,
        width from :py:func:`Gamma2sigma`)

        :param signal: xarray spectral signal to convolve, coord=wl (other dimensions are looped over)
        :param name: name of the output DataArray
        :param info: optional parameter to feed the attributes of the output xarray
        :return: convolved signal on the central wavelengths of the bands
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
