import numpy as np
import xarray as xr

from numba import jit
from scipy import ndimage

import matplotlib.pyplot as plt

from multiprocessing import Pool  # Process pool
from multiprocessing import sharedctypes
import itertools
from scipy.optimize import least_squares, minimize
from dataclasses import dataclass
import os
from dask.distributed import Client, LocalCluster
from dask import compute


def chunk_xarray(obj: xr.DataArray, mask_wv, first_guess, use_x0_near):
    y = obj.transpose("xc", "yc", "wl_sensor").to_numpy()
    mask = np.logical_or(np.all(np.isnan(y), axis=-1), mask_wv)
    if y.shape[0] != 0:
        y = np.ascontiguousarray(y, dtype=np.float64)
    else:
        # dummy array of shape (h, w, 0) to infer output
        y = y
    h, w, c = y.shape
    out = np.full((h, w, len(first_guess)), np.nan)

    # x0 = [20, -0.04, 0.1]
    for ix in range(h):
        for iy in range(w):
            if mask[ix, iy]:
                continue
            y_pix = y[ix, iy]

            xres, std, sucess = self.solve_pix(
                current_x0, y_pix
            )  # TODO: can i pass as kwargs
            out = [*xres, *std]
            if use_x0_near and sucess:
                current_x0 = xres
            out[ix, iy, :] = out

    coords = {
        "xc": obj.xc,
        "yc": obj.yc,
        "inv": [f"param-{str(i)}" for i in range(len(first_guess))],
    }
    output_dataset = xr.DataArray(data=out, coords=coords, dims=["xc", "yc", "inv"])
    return output_dataset


@dataclass
class ProcessMegaPixParameters:
    block_size: int
    Npix_per_megapix: int
    pixel_percentage: float


class Solver:
    def __init__(self, first_guess):
        self.first_guess = first_guess

    def errFit(self, hess_inv, resVariance):
        """
        Error/uncertainty of the estimated parameters
        :param resVariance:
        :return:
        """
        return np.sqrt(np.diag(hess_inv * resVariance))

    def fill_na_conv(self, x):
        """
        Nan-mean convolution
        """
        # get index of central pixel (assume odd size)
        idx = len(x) // 2
        if ~np.isnan(x[idx]):
            return x[idx]
        else:
            return np.nanmean(np.delete(x, idx))

    def fill_na(self, arr, mask=np.ones((3, 3))):
        return ndimage.generic_filter(
            arr,
            function=self.fill_na_conv,
            footprint=mask,
            mode="constant",
            cval=np.nan,
        )

    def conv_mapping(self, x):
        """
        Nan-mean convolution
        """
        # get index of central pixel (assume odd size kernel size)
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
        """
         Function to convolve parameter image with uncertainty image
        :param image: parameter image
        :param weight: uncertainty image
        :param windows: size of the window for convolution
        :return: convolved result with same shape as image

        """
        # TODO (antoine) : check if ix/iy is negative (also iy)?
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
                    # I think that this actually filters out nan valures
                    # if == 0:, results[i,j] = image[i,j]. Coherent ?
                    # other case: local mean weighted by uncertainty
                    for ii in range(Mf):
                        ix = i - Mf2 + ii
                        if ix < M:
                            for jj in range(Nf):

                                iy = j - Nf2 + jj
                                if iy < N:
                                    wgt = weight[ix, iy]
                                    if wgt > 0.0:
                                        num += wgt * image[ix, iy]
                                        norm += wgt
                    result[i, j] = num / norm
        return result

    def solve_pix(self, y_pix):
        raise NotImplementedError("subclass must implement this")

    def localized_average(self, prod, prod_std, windows=np.array([3, 3])):
        prod = prod.to_numpy().astype(float).copy()
        prod_std = prod_std.to_numpy().astype(float).copy()
        weights = 1 / prod_std**2
        return self.filter2d(prod, weights, windows)

    def solve(
        self,
        data: xr.DataArray,
        water_pix_prop,
        process_parameters: ProcessMegaPixParameters,
        dim_na="wl_sensor",
        use_x0_near=False,
    ):
        """
        solve underlying inversion problem per pixel (xc, yc dim)
        args:
            - data: assume input is (yc, xc, wl_sensor)
        output:
            np.array (xc, yc)
        """

        _, height, width = data.shape
        block_size = process_parameters.block_size
        pixel_threshold = process_parameters.pixel_percentage

        number_args = 2 * len(self.first_guess)
        # why inversed height/width?
        result = np.ctypeslib.as_ctypes(np.full((width, height, number_args), np.nan))
        shared_array = sharedctypes.RawArray(result._type_, result)

        global chunk_process
        if water_pix_prop is not None:
            mask = water_pix_prop < pixel_threshold / 100
        else:
            mask = None

        # TODO (antoine): this is not working in python 3.14 because
        # current code rely on fork to work properly.

        def chunk_process(args):

            window_x, window_y = args

            tmp = np.ctypeslib.as_array(shared_array)
            current_x0 = self.first_guess
            # x0 = [20, -0.04, 0.1]
            for ix in range(window_x, min(width, window_x + block_size)):
                for iy in range(window_y, min(height, window_y + block_size)):
                    if mask is not None and mask.isel(xc=ix, yc=iy):
                        continue
                    y_pix = data.isel(xc=ix, yc=iy).dropna(dim=dim_na)

                    xres, std, sucess = self.solve_pix(current_x0, y_pix)
                    out = [*xres, *std]
                    if use_x0_near and sucess:
                        current_x0 = xres
                    tmp[ix, iy, :] = out

            return

        window_idxs = [
            (i, j)
            for i, j in itertools.product(
                range(0, width, block_size), range(0, height, block_size)
            )
        ]

        with Pool() as p:
            res = p.map(chunk_process, window_idxs)

        result = np.ctypeslib.as_array(shared_array)
        return result

    # def solve_xarray(
    #     self,
    #     data: xr.DataArray,
    #     water_pix_prop,
    #     process_parameters: ProcessMegaPixParameters,
    #     dim_na="wl_sensor",
    #     use_x0_near=False,
    # ):
    #     # TODO: check why na in dim_na
    #     # 1. look at non-all na pixels
    #     # 2. A. For thoses pixel is band with na present everywhere
    #     block_size = process_parameters.block_size
    #     pixel_threshold = process_parameters.pixel_percentage
    #     # why inversed height/width?

    #     if water_pix_prop is not None:
    #         mask_wv = water_pix_prop < pixel_threshold / 100
    #     else:
    #         mask_wv = None

    #     input_image = data.rechunk((block_size, block_size, -1))
    #     result = xr.map_blocks(
    #         chunk_xarray,
    #         input_image,
    #         kwargs={
    #             "mask_wv":mask_wv,
    #             "first_guess":self.first_guess,
    #             "use_x0_near": use_x0_near,
    #             "per_pix_func": self.solve_pix
    #         },
    #     )
    #     # serialisation of the function is an issue since the aerosol function contains a big LUT

    #     cluster_kwargs = {
    #         "n_workers": os.cpu_count() - 2,
    #         "threads_per_worker":2,
    #     }
    #     with LocalCluster(**cluster_kwargs) as cluster:
    #         with Client(cluster) as client:
    #             vals = compute(result)
    #     return vals.values


class WaterVapor(Solver):
    def __init__(
        self,
        Twv_lut,
        wl_water_vapor=None,
        # raster_name="coarse_masked_raster",
        # variable="Rtoa",
        first_guess=[2, -0.04, 0.1],
    ):
        super().__init__(first_guess)
        self.Twv_lut = Twv_lut
        # get data for the subset of "water vapor" wavelengths
        self.wl_water_vapor = wl_water_vapor

    def toa_simu(self, wl, Twv, tcwv, a, b):
        """wl in micron"""
        return Twv.interp(tcwv=tcwv, method="linear") * (a * wl + b)

    def toa_simu2(self, wl, Twv, tcwv, c0, c1, c2, c3):

        return c0 * np.exp(-c1 * wl**-c2) * self.Twv_.interp(
            tcwv=tcwv
        ) + c3 * self.wl_**-3 * self.Twv_.interp(tcwv=0.3 * tcwv)

    def func(self, x, Twv, wl, y):
        return self.toa_simu(wl, Twv, *x) - y

    def func2(self, x, Twv, wl, y):
        return self.toa_simu2(wl, Twv, *x) - y

    def solve_pix(self, current_x0, y_pix):

        res_lsq = least_squares(
            self.func,
            current_x0,
            args=(self.Twv_, self.wl_mic, y_pix),
            bounds=([0, -10, 0], [60, 1, 1]),
            diff_step=1e-2,
            xtol=1e-2,
            ftol=1e-2,
            max_nfev=20,
        )

        xres = res_lsq.x
        nb_param = len(xres)
        resVariance = (res_lsq.fun**2).sum() / (len(res_lsq.fun) - len(res_lsq.x))
        hess = np.matmul(res_lsq.jac.T, res_lsq.jac)
        try:
            hess_inv = np.linalg.inv(hess)
            std = self.errFit(hess_inv, resVariance)
        except:
            std = [np.nan for i in range(nb_param)]

        return xres, std, None

    def solve(self, data, water_pix_prop, air_mass_mean, process_parameters):
        x = data.xc
        y = data.yc
        data = data.sel(wl_sensor=self.wl_water_vapor)

        self.air_mass_mean = air_mass_mean
        self.Twv_ = self.Twv_lut.sel(wl_sensor=data.wl_sensor).interp(
            air_mass=air_mass_mean
        )
        wl_mic = self.Twv_["wl_sensor"] / 1000
        self.Twv_["wl_sensor_wv_mic"] = wl_mic
        self.wl_mic = wl_mic.values
        result = super().solve(data, water_pix_prop, process_parameters)
        self.water_vapor = xr.Dataset(
            dict(
                tcwv=(["yc", "xc"], result[:, :, 0].T),
                tcwv_std=(["yc", "xc"], result[:, :, 3].T),
            ),
            coords=dict(xc=np.array(x), yc=np.array(y)),
            attrs=dict(
                description="Fitted Total Columnar Water vapor; warning for transmittance computation only",
                units="kg/m**2",
            ),
        )
        tcwv_smooth = self.localized_average(
            self.water_vapor.tcwv, self.water_vapor.tcwv_std
        )
        tcwv_smooth = self.fill_na(tcwv_smooth, mask=np.ones((5, 5)))
        self.water_vapor["tcwv_smooth"] = (["yc", "xc"], tcwv_smooth)
        return result

    def get_wv_transmittance_raster(self):
        return get_wv_transmittance_raster(
            self.Twv_lut, self.air_mass_mean, self.water_vapor.tcwv_smooth
        )


def get_wv_transmittance_raster(Twv_lut, air_mass_mean, tcwv_smooth):
    """
    Retreive transmittance for water vapor LUT
    """
    # Original was a bit faster:
    # tcwv_vals = self.water_vapor.tcwv.round(1)
    # tcwvs = np.unique(tcwv_vals)
    # tcwvs = tcwvs[~np.isnan(tcwvs)]
    # Twvs = (
    #     self.Twv_lut.Twv.interp(air_mass=self.air_mass_mean)
    #     .interp(tcwv=tcwvs, method="linear")
    #     .drop("air_mass")
    # )
    # return Twvs.interp(tcwv=tcwv_vals, method="nearest")

    t_r = (
        Twv_lut.interp(air_mass=air_mass_mean)
        .interp(tcwv=tcwv_smooth)
        .drop("air_mass")
        .drop("tcwv")
    )

    fill_value = (
        Twv_lut.interp(air_mass=air_mass_mean)
        .interp(tcwv=np.nanmax(tcwv_smooth))
        .drop("air_mass")
        .drop("tcwv")
    )
    return t_r.fillna(fill_value)


def prepare_lut(wl, sensor_desc, aerosol_model, auxdata, raa_lut, aero_lut, pressure):
    sza, vza = (
        sensor_desc.sza_mean,
        sensor_desc.vza_mean,
    )
    auxdata = auxdata

    raa_lut = raa_lut
    sunglint_eps = sensor_desc.sensor_mod.convolve(
        auxdata.sunglint_eps,
    )
    # 2450-2550: 50nm res only for rot. Will fall back to linear integration instead of convolution
    rot = sensor_desc.sensor_mod.convolve(auxdata.rot, fallback_int=[2450, 2550])

    rot = rot * (pressure / auxdata.pressure_rot_ref)

    aot_refs = [0, *np.logspace(-3, np.log10(0.8), 100)]
    aot_lut = aero_lut.sel(model=aerosol_model).aot
    aot_lut = sensor_desc.sensor_mod_lr.convolve(aot_lut)
    aot_lut = aot_lut.interp(aot_ref=aot_refs, method="quadratic").dropna("aot_ref")

    norm_radiance = (
        aero_lut.sel(model=aerosol_model)
        .I.interp(vza=vza, azi=raa_lut, method="linear")
        .interp(sza=sza, method="quadratic")
        .squeeze()
    )
    norm_radiance = sensor_desc.sensor_mod_lr.convolve(norm_radiance)
    Rtoa_lut = (
        norm_radiance.interp(aot_ref=aot_refs, method="quadratic").dropna("aot_ref")
    ) / np.cos(np.radians(sza))
    return (
        sunglint_eps.sel(wl_sensor=wl),
        rot,
        aot_lut.sel(wl_sensor=wl),
        Rtoa_lut.sel(wl_sensor=wl),
    )


def transmittance_dir(aot, M, rot=0):
    return np.exp(-(rot + aot) * M)


def get_aot_diff_trans(aot_ref_smooth, aot_lut, Rtoa_lut, air_mass_mean, rot):
    # TODO (Tristan) update LUT for aot< 0.001
    # # faster version is possible ?
    aot_ref_median = np.nanmedian(aot_ref_smooth.values)
    aot_ref_vals = aot_ref_smooth.fillna(aot_ref_median)
    # aot_ref_vals = aot_ref_vals.round(3)
    # aot_refs = np.unique(aot_ref_vals)
    # aot_refs = aot_refs[~np.isnan(aot_refs)]

    # aot_refs[aot_refs < 0.002] = 0.002

    # # if rounded aot_ref has unique value
    # if len(aot_refs) == 1:
    #     aot_refs = np.concatenate([aot_refs, 1.2 * aot_refs])
    #  aots = self.aot_lut.interp(aot_ref=aot_refs, method="linear")
    # aots = aots.interp(aot_ref=aot_ref_vals, method="nearest")
    # Rdiffs = self.Rtoa_lut.interp(aot_ref=aot_refs, method="linear")
    # Rdiffs = Rdiffs.interp(aot_ref=aot_ref_vals, method="nearest")

    # construct raster for diffuse atmospheric reflectance
    # TODO (tristan) check quadratic interpolation (should be much better)
    aots = aot_lut.interp(aot_ref=aot_ref_vals, method="linear")
    Rdiffs = Rtoa_lut.interp(aot_ref=aot_ref_vals, method="linear")

    # construct raster for direct transmittance due to rayleigh and aerosol
    Tdirs = transmittance_dir(aots, air_mass_mean, rot=rot)
    return aots, Rdiffs, Tdirs


class Aerosol(Solver):

    def __init__(
        self,
        wl_atmo: list[float],
        wl_non_neg: list[float],
        wl_sunglint: list[float],
        pressure=None,
        aero_lut=None,
        Ttot_Ed=None,
        auxdata=None,
        aerosol_model="COAV_rh70",
        first_guess=[0.01, 0],
        aot550_limits=[0.002, 0.8],
    ):
        super().__init__(first_guess)

        self.aerosol_model = aerosol_model
        self.pressure = pressure
        self.auxdata = auxdata
        self.aero_lut = aero_lut
        self.Ttot_Ed = Ttot_Ed

        # set box limits in aod550 for non-linear optimization
        self.aod550_min = aot550_limits[0]
        self.aod550_max = aot550_limits[1]

        # # get full resolution parameters
        # self.xfull = prod.raster.x
        # self.yfull = prod.raster.y

        # get data for the subset of "black water" wavelengths

        self.wl_atmo = wl_atmo
        self.wl_non_neg = wl_non_neg
        self.wl_sunglint = wl_sunglint

    def prepare_lut(self, wl):
        """
        Used for:
        - initial LUT interpolation to wl of sensor
        - redifinition of used wl
        """
        sunglint_eps, rot, aot_lut, Rtoa_lut = prepare_lut(
            wl,
            self.sensor_description,
            self.aerosol_model,
            self.auxdata,
            self.raa_lut,
            self.aero_lut,
            self.pressure,
        )
        self.sunglint_eps = sunglint_eps
        self.rot = rot
        self.aot_lut = aot_lut
        self.Rtoa_lut = Rtoa_lut

    def toa_simu(self, aot, rot, Rtoa_lut, sunglint_eps, aot_ref, BRDFg):
        """ """
        aot = aot.interp(aot_ref=aot_ref)
        Rdiff = Rtoa_lut.interp(aot_ref=aot_ref)
        Tdir = transmittance_dir(aot, self.air_mass_mean, rot=rot)
        sunglint_corr = Tdir * sunglint_eps
        Rdir = (
            sunglint_corr
            * BRDFg
            / (
                Tdir.sel(wl_sensor=self.wl_sunglint)
                * sunglint_eps.sel(wl_sensor=self.wl_sunglint)
            ).mean(dim="wl_sensor")
        )
        # sunglint_toa.Rtoa.plot(x='wl',hue='aot_ref',ax=axs[0])

        return Rdiff + Rdir

    def func(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        return y - self.toa_simu(aot, rot, Rtoa_lut, sunglint_eps, *x)  # /sigma

    def cost_func(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        return np.sum((self.func(x, aot, rot, Rtoa_lut, sunglint_eps, y) ** 2))

    def constraint(self, x, aot, rot, Rtoa_lut, sunglint_eps, y):
        return np.min((self.func(x, aot, rot, Rtoa_lut, sunglint_eps, y)))

    def solve_pix(self, current_x0, y_pix):
        """
        Find aerosol using inversion. For each hyperspectral band,
        transmittance/ glint reflectance / path radiance are approximate by their
        value in the center wavelength
        """
        cons = {
            "type": "ineq",
            "fun": self.constraint,
            "args": (
                self.aot_lut,
                self.rot,
                self.Rtoa_lut,
                self.sunglint_eps,
                y_pix.sel(wl_sensor=self.wl_non_neg, method="nearest"),
            ),
        }
        min_res = minimize(
            self.cost_func,
            current_x0,
            args=(
                self.aot_lut,
                self.rot,
                self.Rtoa_lut,
                self.sunglint_eps,
                y_pix.sel(wl_sensor=self.wl_atmo, method="nearest"),
            ),
            method="SLSQP",
            bounds=((self.aod550_min, self.aod550_max), (0, 1.3)),
            constraints=cons,
            options={"maxiter": 10},
        )
        xres = min_res.x

        std = [min_res.fun, np.sum(min_res.jac**2)]
        return xres, std, min_res.success

    def solve(
        self, data, water_pix_prop, air_mass_mean, sensor_desc, process_parameters
    ):
        data = data
        self.sensor_description = sensor_desc
        self.wl_sensor = sensor_desc.wl_sensor
        self.sza_mean = sensor_desc.sza_mean
        self.vza_mean = sensor_desc.vza_mean
        self.raa_mean = sensor_desc.raa_mean
        self.raa_lut = (180 - self.raa_mean) % 360

        # process parameter
        # self.block_size = self.prod.block_size
        # self.pixel_threshold = self.prod.pixel_threshold
        self.air_mass_mean = air_mass_mean
        self.prepare_lut(data.wl_sensor)

        result = super().solve(
            data, water_pix_prop, process_parameters, use_x0_near=True
        )

        self.aero_img = xr.Dataset(
            dict(
                aot_ref=(["yc", "xc"], result[:, :, 0].T),
                brdfg=(["yc", "xc"], result[:, :, 1].T),
                aot_ref_std=(["yc", "xc"], result[:, :, 2].T),
                brdfg_std=(["yc", "xc"], result[:, :, 3].T),
            ),
            coords=dict(xc=np.array(data.xc), yc=np.array(data.yc)),
            attrs=dict(
                description="aerosol and sunglint retrieval from coarse resolution data",
                aerosol_model=self.aerosol_model,
            ),
        )
        self.smoothing()

    def smoothing(self, windows=np.array([3, 3]), mask=np.ones((3, 3))):
        """
        Replace estimated aot by weighted mean of neighboors values (using retreived std)
        """
        aot_ref_smoothed = self.localized_average(
            self.aero_img["aot_ref"], self.aero_img["aot_ref_std"], windows=windows
        )
        res = ndimage.generic_filter(
            aot_ref_smoothed, function=self.conv_mapping, footprint=mask, mode="nearest"
        )  # TODO (antoine) : there is two smoothing methods

        self.aero_img["aot_ref_smoothed"] = xr.DataArray(
            res,
            coords=dict(yc=np.array(self.aero_img.yc), xc=np.array(self.aero_img.xc)),
        )

    def get_aot_diff_trans(self, wl):

        self.prepare_lut(wl)
        aots, Rdiffs, Tdirs = get_aot_diff_trans(
            self.aero_img.aot_ref_smoothed,
            self.aot_lut,
            self.Rtoa_lut,
            self.air_mass_mean,
            self.rot,
        )
        return aots, Rdiffs, Tdirs

    def compute_atmo_bidir_transmittance(self, use_mean=False):
        if use_mean:
            aot_ref = float(self.aero_img.aot_ref.mean())
        else:
            aot_ref = self.aero_img.aot_ref_smoothed

        Ttot_Ed_opacmodel = self.Ttot_Ed.sel(model=self.aerosol_model)
        sensor_mod = self.sensor_description.sensor_mod_lr
        Ttot_Ed_ = Ttot_Ed_opacmodel.interp(sza=self.sza_mean, method="cubic").interp(
            aot_ref=aot_ref, method="quadratic"
        )
        # fillna -> error when interp != linear
        Ttot_Ed_ = sensor_mod.convolve(Ttot_Ed_.fillna(1))

        Ttot_Lu_ = Ttot_Ed_opacmodel.interp(sza=self.vza_mean, method="cubic").interp(
            aot_ref=aot_ref, method="quadratic"
        )
        Ttot_Lu_ = sensor_mod.convolve(Ttot_Lu_.fillna(1))

        Ttot_Lu_ = Ttot_Lu_**1.05
        Ttot = (Ttot_Ed_ * Ttot_Lu_).reset_coords(drop=True)
        return Ttot

    # def get_aot_full_resolution(self):
    #     self.aot_ref_full = self.aero_img["aot_ref_smoothed"].interp(
    #         x=self.xfull,
    #         y=self.yfull,
    #         method="linear",
    #         kwargs={"fill_value": "extrapolate"},
    #     )
