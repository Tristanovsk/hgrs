import numpy as np
import xarray as xr


from typing import Any
from dataclasses import dataclass
from math import gamma


@dataclass
class SensorDescription:
    wl_sensor: list[float]
    fwhm: list[float]
    sza_mean: float
    vza_mean: float
    raa_mean: float
    sensor_mod: Any
    sensor_mod_lr: Any


def Gamma2sigma(Gamma):
    """Convert FWHM (Gamma) to standard deviation (sigma) for a Gaussian"""
    return Gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def super_gaussian_fwhm2sigma(fwhm, expon):
    """Convert FWHM to sigma for a super-Gaussian (generalized normal). Valid for any exponent b = expon."""
    denum = (
        2 * np.sqrt(gamma(1 / expon) / gamma(3 / expon)) * (np.log(2)) ** (1 / expon)
    )
    return fwhm / denum


def gaussian_sr(x, mu, sigma):
    """Gaussian (normalized to 1 at peak)"""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def super_gaussian_sr(x, mu, sigma, expon):
    alpha = np.sqrt(2) * sigma
    beta = expon
    norm = beta / (2 * alpha * gamma(1 / beta))
    return np.exp(-np.abs((x - mu) / alpha) ** expon) * norm


def gaussian(wl_signal, wl_i, fwhm_i):
    sig = Gamma2sigma(fwhm_i)
    return gaussian_sr(wl_signal, mu=wl_i, sigma=sig)


def super_gaussian(wl_signal, wl_i, fwhm_i, expon=3.0):
    sig = super_gaussian_fwhm2sigma(fwhm_i, expon)
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
        mask_curr = rsr > threshold
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
        return resample_to_target_raster(
            signal, da_rsr, dim_to_resample=dim, fallback_int=fallback_int
        )


class BaselineInterp:
    def __init__(
        self, wl_sensor: np.array, dim_wl_sensor="wl_sensor", inter_mod="linear"
    ):
        self.wl_sensor = np.array(wl_sensor)
        self.dim_wl_sensor = dim_wl_sensor
        self.inter_mod = inter_mod

    def convolve(self, signal, dim="wl"):
        return signal.rename({dim: self.dim_wl_sensor}).interp(
            coords={self.dim_wl_sensor: self.wl_sensor}, method=self.inter_mod
        )


from sklearn.linear_model import LinearRegression


class PrismaSensitivity:
    def __init__(self):
        wls = [
            406.9934,
            415.839,
            423.78476,
            431.3347,
            438.6569,
            446.0147,
            453.38947,
            460.73175,
            468.09845,
            475.31885,
            482.54816,
            489.79483,
            497.0587,
            504.51172,
            512.0464,
            519.54376,
            527.3053,
            535.05255,
            542.88513,
            550.9146,
            559.02026,
            567.2061,
            575.4868,
            583.8441,
            592.339,
            601.0144,
            609.9582,
            618.72,
            627.77844,
            636.6763,
            645.9638,
            655.41876,
            664.8941,
            674.46436,
            684.13727,
            694.12836,
            703.737,
            713.72687,
            723.87994,
            733.9552,
            744.14954,
            754.4696,
            764.85645,
            775.2735,
            785.65955,
            796.127,
            806.71106,
            817.31104,
            827.9195,
            838.5272,
            849.21,
            859.97314,
            870.74255,
            881.45605,
            892.08093,
            902.80164,
            913.44507,
            923.9502,
            934.11206,
            944.6273,
            956.2715,
        ]
        fwhm = np.array(
            [
                11.352248,
                10.377187,
                9.750846,
                9.332583,
                9.22605,
                9.195631,
                9.123976,
                9.059169,
                8.920167,
                8.914632,
                8.908112,
                9.007447,
                9.206349,
                9.2927,
                9.366909,
                9.507074,
                9.641453,
                9.669557,
                9.795126,
                10.017526,
                10.059797,
                10.173916,
                10.31268,
                10.395864,
                10.554132,
                10.826656,
                10.924109,
                11.071507,
                11.219973,
                11.142748,
                11.548293,
                11.6332,
                11.71825,
                11.8385105,
                11.979173,
                12.036037,
                11.982176,
                12.482482,
                12.349199,
                12.419001,
                12.564529,
                12.627702,
                12.805029,
                12.744262,
                12.7641325,
                12.854497,
                12.948172,
                12.956257,
                12.987681,
                13.008858,
                13.088384,
                13.1459,
                13.171485,
                13.043188,
                13.106065,
                13.110408,
                13.066294,
                12.880766,
                12.754054,
                13.277605,
                13.463784,
            ]
        )

        wl_target = np.linspace(350, 1100, 10000)

        g = Gaussian(wls, fwhm)
        l_out = []
        for i in range(len(wls)):
            rsr = g.get_rsr_band(wl_target, i)

            l_out.append(rsr)
        wl_lss = np.array(
            [
                399.1202346,
                409.67741935,
                444.86803519,
                469.50146628,
                495.01466276,
                536.36363636,
                590.02932551,
                637.53665689,
                686.80351906,
                695.60117302,
            ]
        )
        l_ss = np.array(
            [
                0.00000000e00,
                7.24429624e-04,
                1.25128753e-03,
                1.93912980e-03,
                3.27822698e-03,
                3.86362466e-03,
                4.50024463e-03,
                4.75635611e-03,
                5.04173748e-03,
                1.46349419e-05,
            ]
        )
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
        super().__init__(wls)

    def get_rsr_band(self, wl_signal, ii):
        return gaussian(wl_signal, self.wls[ii], self.fwhm[ii])


class SuperGaussian(GenericSpecSen):
    def __init__(self, wls, fwhm, expon=3.0):
        self.wls = wls
        self.fwhm = fwhm
        self.expon = expon
        super().__init__(wls)

    def get_rsr_band(self, wl_signal, ii):
        return super_gaussian(wl_signal, self.wls[ii], self.fwhm[ii], expon=self.expon)


# class Spectral:
#     def __init__(self, central_wl, fwhm, expon=3):
#         """
#         Convolve with spectral response of sensor based on full width at half maximum of each band
#         :param central_wl: numpy array of the central wavelengths
#         :param fwhm: scalar or numpy array containing full width at half maximum in nm
#         :param info: optional parameter to feed the attributes of the output xarray
#         :return:
#         """
#         central_wl = np.array(central_wl).astype(np.float64)
#         fwhm = np.array(fwhm).astype(np.float64)

#         fwhm = xr.DataArray(
#             fwhm,
#             name="fwhm",
#             coords={"wl_sensor": central_wl},
#             attrs={
#                 "definition": "full width at half maximum of spectral responses modeled as gaussian distributions"
#             },
#         )
#         self.fwhm = fwhm
#         self.super_gaussian = SuperGaussian(central_wl, fwhm, expon=expon)
#         self.gaussian = Gaussian(central_wl, fwhm)

#     def convolve2(self, signal):
#         return self.super_gaussian.convolve(signal)

#     def convolve(self, signal):
#         return self.gaussian.convolve(signal)

#     # def plot_rsr(self):
#     # wl_ref = np.linspace(360, 2550, 10000)
#     # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
#     # for fn in self.spectral_sensitivity_fns:
#     #     rsr = fn(wl_ref)
#     #     axs.plot(wl_ref, rsr, "-k", lw=0.5, alpha=0.4)
#     #     axs.set_xlabel("Wavelength (nm)")
#     #     axs.set_ylabel("Spectral response function")
#     # axs.set_xlabel("Wavelength (nm)")
#     # axs.set_ylabel("Spectral response function")

#     # return fig


# fonction: convolve_xarray_to_target_sensor
# args: input_array (xr.DataArray)
# list_spectral_sensitivity_fns: List[Callable (float-> flaot)]
# one of the dimention of the input array is "wl"
# using xarray and the list_spectral_sensitivity_fn, the input xarray is resampled to the target resolution

# from typing import List, Callable
# import xarray as xr
# import numpy as np

# def convolve_xarray_to_target_sensor(
#     input_array: xr.DataArray,
#     list_spectral_sensitivity_fns: List[Callable[[float], float]]
#     sensor_wl: List[float]
# ) -> xr.DataArray:
#     """
#     Convolve an xarray.DataArray with a list of spectral sensitivity functions,
#     handling irregular wavelength spacing and normalizing the output.

#     Args:
#         input_array (xr.DataArray): Input array with a 'wl' dimension.
#         list_spectral_sensitivity_fns (List[Callable]): List of functions mapping wavelength to sensitivity.
#         sensor_wl (List[float]): list of the center wavelength of the given sensor relative spectral response.
#     Returns:
#         xr.DataArray: Resampled DataArray along a new 'sensor' dimension, normalized by sensitivity.
#     """

#     if "wl" not in input_array.dims:
#         raise ValueError("Input array must have a 'wl' dimension")

#     wavelengths = input_array["wl"].values
#     convolved_values = []

#     for fn in list_spectral_sensitivity_fns:
#         # Compute sensitivity for all wavelengths
#         sensitivity = np.array([fn(wl) for wl in wavelengths])

#         # Multiply input by sensitivity
#         weighted = input_array * xr.DataArray(sensitivity, coords={"wl": wavelengths}, dims=["wl"])

#         # Integrate using trapezoidal rule along 'wl'
#         numerator = np.trapezoid(weighted.values, wavelengths, axis=weighted.get_axis_num("wl"))
#         denominator = np.trapezoid(sensitivity, x=wavelengths)
#         normalized = numerator / denominator  # normalize by total sensitivity

#         # Preserve non-wavelength dimensions
#         dims_out = [d for d in input_array.dims if d != "wl"]
#         convolved_values.append(xr.DataArray(normalized, dims=dims_out))

#     # Stack results along new 'sensor' dimension
#     result = xr.concat(convolved_values, dim="sensor_wl")
#     result = result.assign_coords(sensor_wl=sensor_wl)

#     return result
