import numpy as np
import xarray as xr


from math import gamma
import json
from pathlib import Path

from .config import SensorDescription, behavior


def gaussian_fwhm_to_sigma(Gamma):
    """Convert Gaussian FWHM to standard deviation, optionally using legacy math."""
    if behavior.PREVIOUS_BEHAVIOR:
        return Gamma * np.sqrt(2.) / (np.sqrt(2. * np.log(2.)) * 2.)
    return Gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def super_gaussian_fwhm_to_sigma(fwhm, expon):
    """Convert FWHM to sigma using the selected legacy or generalized-normal formula."""
    if behavior.PREVIOUS_BEHAVIOR:
        return fwhm / 2 * (2 * np.log(2)) ** (-1 / expon)
    denum = (
        2 * np.sqrt(gamma(1 / expon) / gamma(3 / expon)) * (np.log(2)) ** (1 / expon)
    )
    return fwhm / denum


def gaussian_sr(x, mu, sigma):
    """Gaussian (normalized to 1 at peak)"""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def super_gaussian_sr(x, mu, sigma, expon):
    if behavior.PREVIOUS_BEHAVIOR:
        sigma = np.maximum(1.e-15, sigma)
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -np.abs(x - mu) ** expon / (2 * sigma ** expon)
        )
    alpha = np.sqrt(2) * sigma
    beta = expon
    norm = beta / (2 * alpha * gamma(1 / beta))
    return np.exp(-np.abs((x - mu) / alpha) ** expon) * norm


def gaussian_spectral_response(wl_signal, wl_i, fwhm_i):
    sig = gaussian_fwhm_to_sigma(fwhm_i)
    return gaussian_sr(wl_signal, mu=wl_i, sigma=sig)


def super_gaussian_spectral_response(wl_signal, wl_i, fwhm_i, expon=3.0):
    sig = super_gaussian_fwhm_to_sigma(fwhm_i, expon)
    return super_gaussian_sr(wl_signal, mu=wl_i, sigma=sig, expon=expon)


def resample_to_target_raster(
    signal,
    da_rsr,
    dim_to_resample="wl",
    dim_tgt="wl_sensor",
    threshold=1e-6,
    fallback_int=[],
):
    list_result = []

    for ii in range(len(da_rsr[dim_tgt])):
        wl_c = da_rsr.wl_sensor[ii].values

        rsr = da_rsr.isel(wl_sensor=ii)
        mask_curr = rsr >= threshold if threshold == 0 else rsr > threshold
        number_valid = np.sum(mask_curr)
        signal_slice = signal.isel(wl=mask_curr)
        rsr_slice = rsr.isel(wl=mask_curr)
        norm = rsr_slice.integrate(coord=dim_to_resample)
        result = (signal_slice * rsr_slice).integrate(coord=dim_to_resample) / norm
        if (
            len(fallback_int) > 0
            and wl_c >= fallback_int[0]
            and wl_c <= fallback_int[1]
        ):
            result = signal.interp(wl=wl_c).rename({"wl": "wl_sensor"})
        elif number_valid <= 3:
            raise ValueError(f"issue when resampling {signal}, {ii}")
        else:
            pass

        list_result.append(result)
    output = xr.concat(list_result, dim=dim_tgt)
    return output


def resample_1d_to_target_raster(signal, rsr, dim_to_resample="wl", threshold=1e-6):

    mask_curr = rsr > threshold
    signal_slice = signal.isel(**{dim_to_resample: mask_curr})
    rsr_slice = rsr.isel(**{dim_to_resample: mask_curr})
    norm = rsr_slice.integrate(coord=dim_to_resample)
    result = (signal_slice * rsr_slice).integrate(coord=dim_to_resample) / norm

    return result


class GenericSpecSen:
    def __init__(self, bands):
        self.bands = bands
        self.number_bands = len(bands)

    def get_rsr_band(self, wl_signal, ii):
        raise NotImplementedError("Subclass must implement this")

    def get_rsr(self, wl_signal, dim="wl"):
        l_rsr = []
        for ii in range(self.number_bands):
            rsr = self.get_rsr_band(wl_signal, ii)

            l_rsr.append(rsr)
        da_rsr = xr.DataArray(l_rsr, coords={"wl_sensor": self.bands, dim: wl_signal})
        return da_rsr

    def convolve(self, signal, dim="wl", fallback_int=[]):
        assert hasattr(signal, dim)
        da_rsr = self.get_rsr(signal[dim], dim=dim)
        threshold = getattr(self, "legacy_threshold", 0.0) if behavior.PREVIOUS_BEHAVIOR else 1e-6
        result = resample_to_target_raster(
            signal, da_rsr, dim_to_resample=dim, threshold=threshold,
            fallback_int=fallback_int
        )
        if behavior.PREVIOUS_BEHAVIOR:
            result = result.astype(np.float32)
        return result


class BaselineInterp:
    def __init__(
        self, wl_sensor: np.array, dim_wl_sensor="wl_sensor", inter_mod="linear"
    ):
        self.wl_sensor = wl_sensor
        self.dim_wl_sensor = dim_wl_sensor
        self.inter_mod = inter_mod

    def convolve(self, signal, dim="wl", wl_sensor=None):
        target_wavelengths = self.wl_sensor if wl_sensor is None else wl_sensor
        signal = (
            signal
            if dim == self.dim_wl_sensor
            else signal.rename({dim: self.dim_wl_sensor})
        )
        return signal.interp(
            {self.dim_wl_sensor: target_wavelengths}, method=self.inter_mod
        )



class PrismaSensitivity:
    def __init__(self):
        from sklearn.linear_model import LinearRegression

        data_path = Path(__file__).with_name("data") / "aux" / "prisma_spectral_sensitivity.json"
        with data_path.open(encoding="utf-8") as data_file:
            spectral_data = json.load(data_file)
        wls = np.asarray(spectral_data["wavelengths_nm"])
        fwhm = np.asarray(spectral_data["fwhm_nm"])
        wl_target = np.linspace(350, 1100, 10000)

        g = Gaussian(wls, fwhm)
        l_out = []
        for i in range(len(wls)):
            rsr = g.get_rsr_band(wl_target, i)

            l_out.append(rsr)
        wl_lss = np.asarray(spectral_data["target_wavelengths_nm"])
        l_ss = np.asarray(spectral_data["target_response"])
        out = xr.DataArray(l_ss, coords=dict(wl=wl_lss)).interp(wl=wl_target).fillna(0)
        X = np.array(l_out).T  # rsr of HyP
        Y = out.values  # / np.max(out.values) # rsr of P

        coefs = LinearRegression(fit_intercept=False, positive=True).fit(X, Y).coef_

        self.coefs = xr.DataArray(coefs, coords=dict(wl=wls))

    def convolve(self, arr, dim="wl"):

        coef_upd = self.coefs.rename({"wl": dim})
        coef_upd[dim] = coef_upd[dim].astype(arr[dim].dtype)
        VNIR = arr.sel(**{dim: coef_upd[dim]}, method="nearest")
        VNIR[dim] = coef_upd[dim]

        return xr.dot(VNIR, coef_upd, dim=dim)


class Gaussian(GenericSpecSen):
    def __init__(self, wls, fwhm):
        self.wls = wls
        self.fwhm = fwhm
        self.legacy_threshold = 0.0
        super().__init__(wls)

    def get_rsr_band(self, wl_signal, ii):
        return gaussian_spectral_response(wl_signal, self.wls[ii], self.fwhm[ii])

    def convolve(self, signal, dim="wl", fallback_int=[], solar_irradiance=False):
        if not solar_irradiance:
            return super().convolve(signal, dim=dim, fallback_int=fallback_int)

        if signal.ndim != 1 or dim not in signal.dims:
            raise ValueError("Solar irradiance convolution expects a 1D spectrum")

        output = []
        for ii in range(self.number_bands):
            fwhm = self.fwhm[ii]
            sigma = fwhm * np.sqrt(2.) / (
                np.sqrt(2. * np.log(2.)) * 2.
            )
            wl_ref = signal[dim]
            rsr = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
                -(wl_ref - self.wls[ii]) ** 2 / (2 * sigma ** 2)
            )
            weighted_integral = (signal * rsr).integrate(dim)
            response_integral = np.trapezoid(rsr, wl_ref)
            output.append(weighted_integral / response_integral)

        return xr.concat(output, dim=xr.DataArray(
            self.bands, dims="wl_sensor", name="wl_sensor"
        ))


class SuperGaussian(GenericSpecSen):
    def __init__(self, wls, fwhm, expon=3.0):
        self.wls = wls
        self.fwhm = fwhm
        self.expon = expon
        self.legacy_threshold = 1e-4
        super().__init__(wls)

    def get_rsr_band(self, wl_signal, ii):
        return super_gaussian_spectral_response(wl_signal, self.wls[ii], self.fwhm[ii], expon=self.expon)
