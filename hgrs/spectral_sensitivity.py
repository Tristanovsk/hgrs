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
    air_mass_mean: float
    sensor_mod: Any
    sensor_mod_lr: Any

def Gamma2sigma(Gamma): 
    """Convert FWHM (Gamma) to standard deviation (sigma) for a Gaussian""" 
    return Gamma / (2.0 * np.sqrt(2.0 * np.log(2.0))) 

def super_gaussian_fwhm2sigma(fwhm, expon): 
    """ Convert FWHM to sigma for a super-Gaussian (generalized normal). Valid for any exponent b = expon. """ 
    denum = 2 * np.sqrt(gamma(1/expon)/gamma(3/expon))* (np.log(2))**(1/expon)
    return fwhm / denum

def gaussian_sr(x, mu, sigma): 
    """ Gaussian (normalized to 1 at peak) """
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma*np.sqrt(2*np.pi))
                  
def super_gaussian_sr(x, mu, sigma, expon): 
    alpha = np.sqrt(2)*sigma
    beta = expon
    norm = beta / (2*alpha * gamma(1/beta))
    return np.exp(-np.abs((x - mu) / alpha)**expon)* norm

def gaussian(wl_signal, wl_i, fwhm_i):
    sig = Gamma2sigma(fwhm_i)
    return gaussian_sr(wl_signal, mu=wl_i, sigma=sig)

def super_gaussian(wl_signal, wl_i, fwhm_i, expon=3.0):
    sig = super_gaussian_fwhm2sigma(fwhm_i, expon)
    return super_gaussian_sr(wl_signal, mu=wl_i, sigma=sig, expon=expon)

def resample_to_target_raster(
    signal, da_rsr, dim_to_resample="wl", dim_tgt="wl_sensor"
):
    list_result = []

    for ii in range(len(da_rsr[dim_tgt])):
        rsr = da_rsr.isel(wl_sensor=ii)
        mask_curr = rsr > 1e-6
        signal_slice = signal.isel(wl=mask_curr)
        rsr_slice = rsr.isel(wl=mask_curr)
        norm = rsr_slice.integrate(coord=dim_to_resample)
        result = (signal_slice * rsr_slice).integrate(coord=dim_to_resample) / norm

        list_result.append(result)
    output = xr.concat(list_result, dim=dim_tgt)
    return output


class GenericSpecSen:
    def __init__(self, bands):
        self.bands = bands
        self.number_bands = len(bands)

    def get_rsr_band(self, wl_signal, ii):
        raise NotImplementedError("Subclass must implement this")

    def get_rsr(self, wl_signal, wl_dim="wl"):
        l_rsr = []
        for ii in range(self.number_bands):
            rsr = self.get_rsr_band(wl_signal, ii)

            l_rsr.append(rsr)
        da_rsr = xr.DataArray(
            l_rsr, coords={"wl_sensor": self.bands, wl_dim: wl_signal}
        )
        return da_rsr

    def convolve(self, signal, wl_dim="wl"):
        assert hasattr(signal, wl_dim)
        da_rsr = self.get_rsr(signal[wl_dim], wl_dim=wl_dim)
        return resample_to_target_raster(signal, da_rsr, dim_to_resample=wl_dim)


class BaselineInterp:
    def __init__(
        self, wl_sensor: np.array, dim_wl_sensor="wl_sensor", inter_mod="linear"
    ):
        self.wl_sensor = np.array(wl_sensor)
        self.dim_wl_sensor = dim_wl_sensor
        self.inter_mod = inter_mod

    def convolve(self, signal, wl_dim="wl"):
        return signal.rename({wl_dim: self.dim_wl_sensor}).interp(
            coords={self.dim_wl_sensor: self.wl_sensor}, method=self.inter_mod
        )


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


class Spectral:
    def __init__(self, central_wl, fwhm, expon=3):
        """
        Convolve with spectral response of sensor based on full width at half maximum of each band
        :param central_wl: numpy array of the central wavelengths
        :param fwhm: scalar or numpy array containing full width at half maximum in nm
        :param info: optional parameter to feed the attributes of the output xarray
        :return:
        """
        central_wl = np.array(central_wl).astype(np.float64)
        fwhm = np.array(fwhm).astype(np.float64)

        fwhm = xr.DataArray(
            fwhm,
            name="fwhm",
            coords={"wl_sensor": central_wl},
            attrs={
                "definition": "full width at half maximum of spectral responses modeled as gaussian distributions"
            },
        )
        self.fwhm = fwhm
        self.super_gaussian = SuperGaussian(central_wl, fwhm, expon=expon)
        self.gaussian = Gaussian(central_wl, fwhm)

    def convolve2(self, signal):
        return self.super_gaussian.convolve(signal)

    def convolve(self, signal):
        return self.gaussian.convolve(signal)

    # def plot_rsr(self):
    # wl_ref = np.linspace(360, 2550, 10000)
    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    # for fn in self.spectral_sensitivity_fns:
    #     rsr = fn(wl_ref)
    #     axs.plot(wl_ref, rsr, "-k", lw=0.5, alpha=0.4)
    #     axs.set_xlabel("Wavelength (nm)")
    #     axs.set_ylabel("Spectral response function")
    # axs.set_xlabel("Wavelength (nm)")
    # axs.set_ylabel("Spectral response function")

    # return fig


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
