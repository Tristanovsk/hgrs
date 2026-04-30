import os
import yaml

import numpy as np


import matplotlib.pyplot as plt


from omnicloudmask import predict_from_array
import hgrs
from hgrs.solver import ProcessMegaPixParameters, WaterVapor, Aerosol
from . import AuxData
import datetime as dt
from dataclasses import dataclass
from hgrs.spectral_sensitivity import SensorDescription
import xarray as xr

from hgrs.lut_tables import LUTTables
from hgrs.cams_data import CAMSProduct
from hgrs.spectral_sensitivity import SensorDescription

opj = os.path.join


class CAMSProduct:

    def __init__(self, path, date, clat, clon, aero_lut):
        """
        Initialise the CAMS data:

        Attributes:
            pressure (float): from CAMS
            to3c (float): from CAMS
            tno2c (float): from CAMS
            tch4c (float): from CAMS
            aod550:
            aerosol_model: closest opac model (using in situ aod at different wl)

        """

        # lazy loading
        cams = xr.open_dataset(
            path, decode_cf=False, chunks={"time": 1, "x": 500, "y": 500}
        )

        cams["forecast_period"].attrs.pop("dtype", None)
        cams = xr.decode_cf(cams)

        # fix for new ADS format (sept 2024)
        if ("forecast_period" in cams.dims) & ("forecast_reference_time" in cams.dims):
            cams = (
                cams.stack(time_buffer=["forecast_period", "forecast_reference_time"])
                .swap_dims({"time_buffer": "valid_time"})
                .sortby("valid_time")
                .rename({"valid_time": "time"})
                .drop_vars(["time_buffer"])
            )

        # slicing
        cams = cams.sel(time=date, method="nearest")
        cams = cams.sel(latitude=clat, longitude=clon, method="nearest")

        # select OPAC aerosol model
        # aod = cams[['aod355', 'aod380', 'aod400', 'aod440', 'aod469', 'aod500', 'aod550', 'aod645', 'aod670',
        #            'aod800', 'aod865', 'aod1020', 'aod1064', 'aod1240', 'aod1640', 'aod2130']].to_pandas()
        # aod.index = aod.index.str.replace('aod', '').astype(int)
        # cams_aod = aod.to_xarray().rename({'index': 'wl'})
        cams_wls = [469, 550, 670, 865, 1240]
        param_aod = []
        for wl in cams_wls:
            wl_ = str(wl)
            param_aod.append("aod" + wl_)

        cams_aod = cams[param_aod].to_array(dim="wl")

        wl_cams = cams_aod.wl.str.replace("aod", "").astype(float)
        cams_aod = cams_aod.assign_coords(wl=wl_cams)

        # new LUT:
        lut_aod = aero_lut.aot.sel(aot_ref=1).interp(wl=cams_aod.wl)
        idx = np.abs((cams_aod / cams.aod550) - lut_aod).sum("wl").argmin()
        aerosol_model = aero_lut.model.values[idx]

        # set gases and pressure
        self.pressure = float(cams.sp) * 1e-2

        self.to3c = float(cams.gtco3)
        self.tno2c = float(cams.tcno2)
        self.tch4c = float(cams.tc_ch4)
        self.aod550 = cams.aod550
        self.aerosol_model = aerosol_model


@dataclass
class WaterMaskParams:
    sunglint_threshold: float
    ndwi_threshold: float
    green_swir_index_threshold: float
    wl_1600: slice
    wl_green: slice
    wl_nir: slice


class Product:
    def __init__(
        self,
        l1c_obj=None,
        cams: CAMSProduct = None,
        lut_tables: LUTTables = None,
        sensor_description: SensorDescription = None,
        xcoarsen=20,
        ycoarsen=20,
        expon=2,
    ):

        self.raster = l1c_obj.copy()
        self.sensor_description = sensor_description
        # spectral parameters
        self.wl_water_vapor = slice(800, 1300)
        self.wl_sunglint = slice(2150, 2250)
        # self.wl_atmo = slice(950, 2450)
        self.wl_atmo = [
            1000,
            1050,
            1075,
            1100,
            1200,
            1300,
            1600,
            1650,
            1700,
            2150,
            2200,
            2250,
        ]
        self.wl_non_neg = [419.457, 490, 560, 650, 750, 800, 865, 1650, 2250]
        self.wl_to_remove = [
            (935, 967),
            (1105, 1170),
            (1320, 1490),
            (1778, 2033),
            (2465, 2550),
        ]

        self.wl_rgb = [30, 20, 10]

        # coarsening parameters
        self.xcoarsen = xcoarsen
        self.ycoarsen = ycoarsen
        Npix_per_megapix = self.xcoarsen * self.ycoarsen
        block_size = 2

        # minimum percentage of water pixel within the mega-pixel to enable processing
        pixel_percentage = 20
        # pixel_threshold = pixel_percentage / 100 * Npix_per_megapix
        self.process_parameters = ProcessMegaPixParameters(
            block_size=block_size,
            Npix_per_megapix=Npix_per_megapix,
            pixel_percentage=pixel_percentage,
            # pixel_threshold=pixel_threshold,
        )

        # number of digits to keep for angle values
        self.ang_resol = 1  # (degree)

        # lut:
        self.lut_tables = lut_tables
        self.gas_lut = lut_tables.gas_lut

        # mask thresholding parameters
        sunglint_threshold = 0.11
        ndwi_threshold = 0.01
        green_swir_index_threshold = 0.1
        wl_green = slice(540, 570)
        wl_nir = slice(850, 882)
        #
        wl_1600 = slice(1580, 1650)
        self.water_threshold_params = WaterMaskParams(
            sunglint_threshold,
            ndwi_threshold,
            green_swir_index_threshold,
            wl_1600=wl_1600,
            wl_green=wl_green,
            wl_nir=wl_nir,
        )

        # atmosphere auxiliary data
        self.cams_data = cams
        self.aerosol_model = cams.aerosol_model

        self.coef_abs_scat = 1.0

        aod550_mean = cams.aod550.mean().values
        # TODO (Tristan) double check regularization from CAMS AOT values
        # TODO: (Antoine): the retreived cams.aod550 is a single values, and not mean over studied area. Can this impact bounds?

        # initialisation of the aerosol retreival
        aod550_std = cams.aod550.std().values
        aod550_std = np.max([aod550_std, 0.2 * aod550_mean + 0.05])
        self.aod550_mean = aod550_mean
        self.aod550_std = aod550_std
        self.aot550_min = 0.002  # np.max([aod550_mean - 2*aod550_std,0.001])
        self.aot550_max = self.aod550_mean + 2 * self.aod550_std

        self.Tg_other = None
        # get hgrs auxdata
        self.auxdata = AuxData(self.sensor_description.sensor_mod)

        # spectral function for sensor response convolution
        # exponent of the super-gaussian spectral response function
        self.expon = expon
        # set the convolution module

    def get_ndwi(self):
        green = self.raster.Rtoa.sel(
            wl_sensor=self.water_threshold_params.wl_green
        ).mean(dim="wl_sensor")
        nir = self.raster.Rtoa.sel(wl_sensor=self.water_threshold_params.wl_nir).mean(
            dim="wl_sensor"
        )
        self.ndwi = (green - nir) / (green + nir)

    def get_green_swir_index(self):
        green = self.raster.Rtoa.sel(
            wl_sensor=self.water_threshold_params.wl_green
        ).mean(dim="wl_sensor")
        b1600 = self.raster.Rtoa.sel(
            wl_sensor=self.water_threshold_params.wl_1600
        ).mean(dim="wl_sensor")
        self.green_swir_index = (green - b1600) / (green + b1600)

    def get_b2200(self):
        self.b2200 = self.raster.Rtoa.sel(wl_sensor=self.wl_sunglint).mean(
            dim="wl_sensor"
        )

    def apply_water_masks(self):
        self.get_ndwi()
        self.get_green_swir_index()
        self.get_b2200()
        self.raster["Rtoa"] = (
            self.raster.Rtoa.where(
                self.ndwi > self.water_threshold_params.ndwi_threshold
            )
            .where(self.b2200 < self.water_threshold_params.sunglint_threshold)
            .where(
                self.green_swir_index
                > self.water_threshold_params.green_swir_index_threshold
            )
            .load()
        )

    def get_omnicloudmask(self, rgnir):
        """
        Apply OmniCloudMAsk for clouds and cloud shadows masking

        Outputs:
            0 = Clear
            1 = Thick Cloud
            2 = Thin Cloud
            3 = Cloud Shadow

        see https://github.com/DPIRD-DMA/OmniCloudMask

        refs:
         Wright, N., Duncan, J. M. A., Callow, J. N., Thompson, S. E., & George, R. J. (2025).
         Training sensor-agnostic deep learning models for remote sensing:
         Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask.
         Remote Sensing of Environment, 322, 114694. https://doi.org/10.1016/J.RSE.2025.114694

        :param rgnir: raster xarray object with the red, green and nir bands
        :return omnimask: raster of the retrieved mask
        """

        pred = predict_from_array(rgnir.fillna(0).values)
        omnimask = xr.DataArray(
            pred[0],
            dims=["y", "x"],
            coords=dict(
                x=rgnir.x.values,
                y=rgnir.y.values,
                time=rgnir.time,
            ),
            attrs=dict(
                description="OmniCloudMask, see https://github.com/DPIRD-DMA/OmniCloudMask",
                reference="https://doi.org/10.1016/J.RSE.2025.114694",
            ),
        )
        omnimask.name = "omnimask"
        return omnimask

    def round_angles(self):
        for param in ["sza", "vza", "raa"]:
            self.raster[param] = self.raster[param].round(self.ang_resol)

    # @staticmethod
    # def remove_wl_dataarray(xarr, wl_to_remove, drop=True):
    #     xarr_ = xarr.isel(x=1, y=1)
    #     for wls in wl_to_remove:
    #         wl_min, wl_max = wls
    #         xarr_ = xarr_.where(
    #             (xarr_.wl_sensor < wl_min) | (xarr_.wl_sensor > wl_max), drop=drop
    #         )
    #     wl_final = xarr_.wl_sensor.values
    #     return xarr.sel(wl_sensor=wl_final)

    @staticmethod
    def remove_wl_dataset(xds, wl_to_remove, variable="Rtoa", drop=True):
        raster = xds[variable]
        if hasattr(raster, "x"):
            coord = dict(x=1, y=1)
        else:
            coord = dict(xc=1, yc=1)
        xarr_ = raster.isel(**coord)
        for wls in wl_to_remove:
            wl_min, wl_max = wls
            xarr_ = xarr_.where(
                (xarr_.wl_sensor < wl_min) | (xarr_.wl_sensor > wl_max), drop=drop
            )
        wl_final = xarr_.wl_sensor.values
        return xds.sel(wl_sensor=wl_final)

    # ------------------- PLOTS ---------------------

    def plots(
        self,
        params,
        titles,
        nrows=1,
        ncols=4,
        figsize=(20, 4),
        cmap=plt.cm.Spectral_r,
        adjust_subplots=False,
        kwargs={},
    ):

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if adjust_subplots:
            fig.subplots_adjust(
                bottom=0.1, top=0.95, left=0.1, right=0.99, hspace=0.15, wspace=0.15
            )
        for i, ax in enumerate(axs.ravel()):
            if isinstance(kwargs, list):
                c_kwargs = kwargs[i]
            else:
                c_kwargs = kwargs

            params[i].plot.imshow(ax=ax, robust=True, cmap=cmap, **c_kwargs)
            ax.set_title(titles[i])
            ax.set(xticks=[], yticks=[])
            ax.set_ylabel("")
            ax.set_xlabel("")
        return fig

    def plot_angles(
        self, raster_name="raster", figsize=(20, 4), cmap=plt.cm.Spectral_r, **kwargs
    ):
        raster = self.__dict__[raster_name]
        params = [raster.sza, raster.vza, raster.raa, raster.air_mass]
        titles = ["SZA", "VZA", "rel. AZI", "Air mass"]
        return self.plots(params, titles, nrwos=1, ncols=4, cmap=cmap, figsize=figsize)

    def plot_params(
        self,
        xds,
        params=["aot_ref", "aot_ref_std", "brdfg", "brdfg_std"],
        shrink=0.8,
        cmap=plt.cm.Spectral_r,
    ):
        titles = params
        params = [xds[params[i]] for i in range(len(params))]

        kwargs = [
            {
                "vmin": 0,
                "cbar_kwargs": {"shrink": shrink, "label": param},
            }
            for param in titles
        ]
        ncols = len(params)
        fig_width = ncols * 5 + 2
        self.plots(params, titles, 1, ncols, (fig_width, 4), cmap=cmap, kwargs=kwargs)

    def plot_masks(
        self,
        params=["cloud_mask", "sunglint_mask", "landcover_mask"],
        vmax=12,
        shrink=0.8,
        cmap=plt.cm.Spectral_r,
    ):
        titles = params
        params = [self.raster[params[i]] for i in range(len(params))]

        ncols = len(params)
        fig_width = ncols * 5 + 1
        kwargs = {
            "vmax": vmax,
            "cbar_kwargs": {"shrink": shrink},
        }
        return self.plots(
            params,
            titles,
            1,
            ncols,
            (fig_width, 4),
            adjust_subplot=True,
            cmap=cmap,
            kwargs=kwargs,
        )

    def plot_rgb(
        self,
        variable="Rtoa",
        raster_name="raster",
        gamma=0.5,
        brightness_factor=1,
        **kwargs,
    ):
        fig = (
            self.__dict__[raster_name][variable].isel(wl_sensor=self.wl_rgb) ** gamma
            * brightness_factor
        ).plot.imshow(rgb="wl_sensor", robust=True, **kwargs)
        fig.axes.set(xticks=[], yticks=[])
        fig.axes.set_ylabel("")
        fig.axes.set_xlabel("")
        return fig

    def plot_water_pix_prop(self, cmap=plt.cm.Spectral_r, **kwargs):
        try:
            fig = self.coarse_masked_raster["water_pix_prop"].plot.imshow(
                cmap=cmap, robust=True, **kwargs
            )
            fig.axes.set(xticks=[], yticks=[])
            fig.axes.set_ylabel("")
            fig.axes.set_xlabel("")
            return fig
        except:
            print("please apply algo.get_coarse_masked_raster() before")


class Algo(Product):
    """Main object for atmospheric correction"""

    def get_pressure(self, alt, psl):
        """Compute the pressure for a given altitude
        alt : altitude in meters (float or np.array)
        psl : pressure at sea level in hPa
        palt : pressure at the given altitude in hPa"""

        palt = psl * (1.0 - 0.0065 * np.nan_to_num(alt) / 288.15) ** 5.255
        return palt

    def get_coarse_masked_raster(
        self,
        raster_name_orig="raster",
        raster_name_coarse="coarse_masked_raster",
        variables=["sza", "vza", "raa", "air_mass", "Rtoa"],
    ):
        # xc -> x position of coarse pixel
        raster_out = (
            self.__dict__[raster_name_orig][variables]
            .coarsen(x=self.xcoarsen, y=self.ycoarsen, boundary="pad")
            .mean()
            .rename({"x": "xc", "y": "yc"})
        )
        water_pixel_number = (
            self.raster["Rtoa"]
            .isel(wl_sensor=slice(10, 20))
            .mean(dim="wl_sensor")
            .coarsen(x=self.xcoarsen, y=self.ycoarsen, boundary="pad")
            .count()
            .rename({"x": "xc", "y": "yc"})
        )
        water_pix_prop = water_pixel_number / self.process_parameters.Npix_per_megapix

        water_pix_prop.name = "water_pix_prop"

        raster_out["water_pix_prop"] = water_pix_prop
        setattr(self, raster_name_coarse, raster_out)

    # def get_full_resolution(self, xarr):
    #     return xarr.interp(xc=self.raster.x, yc=self.raster.y).drop_vars(["xc","yc"])

    # ------------ OTHER gazes --------------

    def get_gaseous_transmittance(self, air_mass_mean):

        gas_lut = self.gas_lut

        ot_o3 = gas_lut.o3 * self.cams_data.to3c
        ot_ch4 = gas_lut.ch4 * self.cams_data.tch4c
        ot_no2 = gas_lut.no2 * self.cams_data.tno2c
        ot_air = (
            (
                gas_lut.co
                + self.coef_abs_scat * gas_lut.co2
                + self.coef_abs_scat * gas_lut.o2
                + self.coef_abs_scat * gas_lut.o4
            )
            * self.cams_data.pressure
            / 1000
        )
        self.abs_gas_opt_thick = ot_ch4 + ot_no2 + ot_o3 + ot_air
        Tg = np.exp(-air_mass_mean * self.abs_gas_opt_thick)
        self.Tg_other = self.sensor_description.sensor_mod.convolve(
            Tg
        )  # name="Ttot", copy_info=False

    def other_gas_correction(self, raster_name="coarse_masked_raster", variable="Rtoa"):
        raster = self.__dict__[raster_name]
        air_mass_mean = np.nanmean(self.__dict__[raster_name]["air_mass"])
        self.get_gaseous_transmittance(air_mass_mean)

        attrs = raster[variable].attrs
        if attrs.__contains__("other_gas_correction") and attrs["other_gas_correction"]:
            print(
                "raster "
                + raster_name
                + "."
                + variable
                + " is already corrected for other gases transmittance"
            )
            print("set attribute other_gas_correction to False to proceed anyway")
            return

        raster[variable] = raster[variable] / self.Tg_other
        raster[variable].attrs["other_gas_correction"] = True

    # ----------- Water vapor correction -----------
    def drop_band_high_gaz_absorption(self, raster_name, variable="Rtoa"):
        Tg_tot = self.Tg_other * self.Twv_raster.mean(["xc", "yc"])

        # remove bands where Tg is below a threshold (typically Tg <= 0.5)
        raster = self.__dict__[raster_name]
        raster[variable] = raster[variable].where(Tg_tot > 0.5, drop=True)

    def drop_band_predefined(self, raster_name, variable):

        raster = self.__dict__[raster_name]
        raster_band_remove = self.remove_wl_dataset(
            raster, self.wl_to_remove, variable=variable
        )
        setattr(self, raster_name, raster_band_remove)

    def find_coarse_water_vapor(
        self, raster_name="coarse_masked_raster", variable="Rtoa"
    ):
        # raster_name="coarse_masked_raster",
        # variable="Rtoa",
        data = self.__dict__[raster_name][variable]
        air_mass_mean = np.nanmean(self.__dict__[raster_name]["air_mass"])
        wv_retrieval = WaterVapor(
            Twv_lut=self.lut_tables.Twv_lut,
            wl_water_vapor=self.wl_water_vapor,
        )

        wv_retrieval.solve(
            data=data,
            water_pix_prop=self.derive_water_pixel_prop(raster_name),
            air_mass_mean=air_mass_mean,
            process_parameters=self.process_parameters,
        )
        self.wv_retrieval = wv_retrieval

    def coarse_water_vapor_correction(
        self, raster_name="coarse_masked_raster", variable="Rtoa"
    ):
        """

        :param raster_name:
        :param variable:
        :return:
        """
        self.Twv_raster = self.wv_retrieval.get_wv_transmittance_raster()
        raster = self.__dict__[raster_name]
        attrs = raster[variable].attrs
        if (
            attrs.__contains__("water_vapor_correction")
            and attrs["water_vapor_correction"]
        ):
            print(
                "raster "
                + raster_name
                + "."
                + variable
                + " is already corrected for water vapor transmittance"
            )
            print("set attribute other_gas_correction to False to proceed anyway")
            return

        raster[variable] = raster[variable] / self.Twv_raster
        raster[variable].attrs["water_vapor_correction"] = True

    # ------ Aerosols -------
    def find_coarse_aeroglint(self):
        raster_name = "coarse_masked_raster"
        variable = "Rtoa"
        raster = self.__dict__[raster_name]
        air_mass_mean = np.nanmean(self.__dict__[raster_name]["air_mass"])

        aero_retrieval = Aerosol(
            wl_atmo=self.wl_atmo,
            wl_non_neg=self.wl_non_neg,
            wl_sunglint=self.wl_sunglint,
            pressure=self.cams_data.pressure,
            aero_lut=self.lut_tables.aero_lut,
            auxdata=self.auxdata,
            Ttot_Ed=self.lut_tables.Ttot_Ed,
            aerosol_model=self.cams_data.aerosol_model,
            first_guess=[self.aod550_mean, 0.0],
            aot550_limits=[self.aot550_min, self.aot550_max],
        )  # state (initialize variables)

        aero_retrieval.solve(
            raster[variable],
            raster["water_pix_prop"],
            air_mass_mean=air_mass_mean,
            sensor_desc=self.sensor_description,
            process_parameters=self.process_parameters,
        )  # state (create aero_img)
        self.aero_retrieval = aero_retrieval

    def compute_atmo_img_smooth(self, wl):

        aots, Rdiffs, Tdirs = self.aero_retrieval.get_aot_diff_trans(wl)
        # define the aot_ref_smoothed (used for the subsequent processing)
        # self.aero_retrieval.get_aot_full_resolution()

        # construct aot raster
        # create a LUT with value of aot @ref wl rounded to 3 significative number (interpolation of original lut)
        # apply the LUT to the data (ie)

        aots.name = "aot"
        aots.attrs["description"] = "spectral aerosol optical thickness"

        Rdiffs.name = "Rtoa_diff"
        Rdiffs.attrs["description"] = "top-of-atmosphere atmosphere reflectance"

        Tdirs.name = "Tdir"
        Tdirs.attrs["description"] = (
            "direct transmittance due to rayleigh and aerosol for total air mass"
        )

        # merge into dataset
        self.atmo_img = xr.merge([aots, Rdiffs, Tdirs])
        self.atmo_img.attrs["description"] = (
            "atmospheric parameters for rayleigh and aerosol components",
        )
        self.atmo_img.attrs["aerosol_model"] = self.aerosol_model

    def compute_atmo_bidir_transmittance(self):
        return self.aero_retrieval.compute_atmo_bidir_transmittance()

    # -------- Full resolution ------------
    def fullres_water_vapor_correction(
        self, raster_name="raster", variable="Rtoa", raster_wv="Twv_raster", chunk=256
    ):

        raster = self.__dict__[raster_name]
        var = raster[variable]
        height, width, Nwl = len(raster.y), len(raster.x), len(raster.wl_sensor)
        # results = np.full((height, width), 0, dtype=np.float32)
        twv_coarse = self.__dict__[raster_wv]
        # TODO (Antoine): resample twv_coarse for panchromatic usage
        # TODO (Antoine): with chunking dask?
        variable = "Rtoa"
        for iy in range(0, height, chunk):
            yc = min(height, iy + chunk)
            if yc > height:
                continue
            for ix in range(0, width, chunk):
                xc = min(width, ix + chunk)
                if xc > width:
                    continue
                tgt_var = var[:, iy:yc, ix:xc]
                Twv_fullres = twv_coarse.interp(xc=tgt_var.x, yc=tgt_var.y).drop_vars(
                    ["xc", "yc"]
                )
                raster[variable].data[:, iy:yc, ix:xc] = tgt_var / Twv_fullres

    def fullres_aeroglint_correction(
        self, raster_name="raster", variable_hyp="Rtoa", chunk=256
    ):
        raster = self.__dict__[raster_name]

        height, width, Nwl = len(raster.y), len(raster.x), len(raster.wl_sensor)

        # retreived params
        Rdiff_full = (
            self.atmo_img.Rtoa_diff
        )  # .interp(x=prod.raster.x, y=prod.raster.y)
        Tdir_full = self.atmo_img.Tdir  # .interp(x=prod.raster.x, y=prod.raster.y)
        sunglint_eps = self.aero_retrieval.sunglint_eps

        wl_sunglint = self.wl_sunglint

        BRDF_sunglint = np.full((height, width), np.nan, dtype=np.float32)
        Rrs = np.full((Nwl, height, width), np.nan, dtype=np.float32)
        # TODO (antoine): resampling of apparent glint/ transmittance
        # TODO (antoine): resampling of apparent glint
        # TODO (antoine): with chunking dask
        for iy in range(0, height, chunk):
            yc = min(height, iy + chunk)
            if yc > height:
                continue
            for ix in range(0, width, chunk):
                xc = min(width, ix + chunk)
                if xc > width:
                    continue
                # correct for diffuse path radiance
                Rcorr = raster[variable_hyp][:, iy:yc, ix:xc]
                Rdiff_full_ = Rdiff_full.interp(xc=Rcorr.x, yc=Rcorr.y).drop_vars(
                    ["xc", "yc"]
                )
                Rcorr = Rcorr - Rdiff_full_

                Tdir_full_ = Tdir_full.interp(xc=Rcorr.x, yc=Rcorr.y).drop_vars(
                    ["xc", "yc"]
                )
                BRDF_sunglint_ = (
                    Rcorr.sel(wl_sensor=wl_sunglint)
                    / (
                        Tdir_full_.sel(wl_sensor=wl_sunglint)
                        * sunglint_eps.sel(wl_sensor=wl_sunglint)
                    )
                ).mean(dim="wl_sensor")
                BRDF_sunglint[iy:yc, ix:xc] = BRDF_sunglint_
                sunglint_app_ = Tdir_full_ * sunglint_eps

                Rdir = sunglint_app_ * BRDF_sunglint_
                Rrs[:, iy:yc, ix:xc] = (Rcorr - Rdir) / np.pi
        return Rrs, BRDF_sunglint

    # --- Export function -----
    # produce ancilliary data
    def derive_water_pixel_prop(self, name="coarse_masked_raster"):
        assert hasattr(self, name), f"{name} does not exist"
        raster = self.__dict__[name]
        assert hasattr(
            raster, "water_pix_prop"
        ), f"{name} does not have a water_pix_prop property"
        return self.coarse_masked_raster.water_pix_prop

    def complete_L2A_attributes(
        self,
        l2_prod: xr.Dataset,
    ):
        """
        Complete the metadata of xarray with correction atmospheric related parameters
        """

        keys = [
            "wl_water_vapor",
            "wl_sunglint",
            "wl_atmo",
            "wl_to_remove",
            "wl_green",
            "wl_nir",
            "wl_1600",
            "wl_rgb",
            "xcoarsen",
            "ycoarsen",
            "Npix_per_megapix",
            "block_size",
            "pixel_percentage",
            # "pixel_threshold",
            "ang_resol",
            "abs_gas_file",
            "lut_file",
            "sunglint_threshold",
            "ndwi_threshold",
            "green_swir_index_threshold",
            "pressure",
            "to3c",
            "tno2c",
            "tch4c",
            "coef_abs_scat",
        ]
        dict_attrs = self.__dict__
        dict_attrs = dict_attrs.copy()
        dict_attrs.update(self.sensor_description.__dict__.copy())
        dict_attrs.update(self.process_parameters.__dict__.copy())
        dict_attrs.update(self.water_threshold_params.__dict__.copy())
        dict_attrs.update(self.cams_data.__dict__.copy())
        dict_attrs.update(self.lut_tables.__dict__.copy())

        aerosol_model = self.aerosol_model
        param = "Rrs"
        l2_prod[param].attrs["unit"] = "per steradian"
        l2_prod[param].attrs["long_name"] = "Remote sensing reflectance"
        l2_prod[param].attrs["description"] = (
            "Directional water-leaving radiance normalized "
            + "by downwelling irradiance in the observation geometry"
        )

        param = "water_pix_prop"
        l2_prod[param].attrs["unit"] = "-"
        l2_prod[param].attrs[
            "description"
        ] = "Relative number of water pixel within mega-pixel used for inversion"

        param = "brdfg"
        l2_prod[param].attrs["unit"] = "-"
        l2_prod[param].attrs["long_name"] = "BRDF_sunglint"
        l2_prod[param].attrs["description"] = (
            "Bidirectional reflectance distribution function "
            + "estimated from the sunglint in the SWIR for the observation geometry"
        )
        param = "brdfg_std"
        l2_prod[param].attrs["unit"] = "-"
        l2_prod[param].attrs["long_name"] = "BRDF_sunglint_standard deviation"
        l2_prod[param].attrs[
            "description"
        ] = "Uncertainty based on optimal estimation procedure"
        param = "brdfg_full"
        l2_prod[param].attrs["unit"] = "-"
        l2_prod[param].attrs["long_name"] = "BRDF_sunglint"
        l2_prod[param].attrs["description"] = (
            "Bidirectional reflectance distribution function "
            + "estimated from the sunglint in the SWIR for the observation geometry"
        )

        param = "aot_ref"
        l2_prod[param].attrs["unit"] = "-"
        l2_prod[param].attrs["long_name"] = "aerosol_optical_thickness"
        l2_prod[param].attrs[
            "description"
        ] = "Aerosol optical thickness at the reference wavelength (550nm)"
        param = "aot_ref_std"
        l2_prod[param].attrs["unit"] = "-"
        l2_prod[param].attrs[
            "long_name"
        ] = "aerosol_optical_thickness_standard_deviation"
        l2_prod[param].attrs[
            "description"
        ] = "Uncertainty based on optimal estimation procedure"
        # param = 'aot_ref_full'
        # l2_prod[param].attrs['unit'] = '-'
        # l2_prod[param].attrs['long_name'] = 'aerosol_optical_thickness'
        # l2_prod[param].attrs['description'] = 'Aerosol optical thickness at the reference wavelength (550nm)'

        param = "tcwv"
        l2_prod[param].attrs["unit"] = "kg m-2"
        l2_prod[param].attrs["long_name"] = "total_columnar_water_vapor"
        l2_prod[param].attrs[
            "description"
        ] = "Water vapor integrated over the atmospheric layer"
        param = "tcwv_std"
        l2_prod[param].attrs["unit"] = "kg m-2"
        l2_prod[param].attrs[
            "long_name"
        ] = "total_columnar_water_vapor_standard_deviation"
        l2_prod[param].attrs[
            "description"
        ] = "Uncertainty based on optimal estimation procedure"
        # param = 'tcwv_full'
        # l2_prod[param].attrs['unit'] = 'kg m-2'
        # l2_prod[param].attrs['long_name'] = 'total_columnar_water_vapor'
        # l2_prod[param].attrs['description'] = 'Water vapor integrated over the atmospheric layer'

        l2_prod["pressure"] = self.cams_data.pressure
        l2_prod["pressure"].attrs["unit"] = "hPa"
        l2_prod["pressure"].attrs[
            "description"
        ] = "Atmospheric pressure at the surface level"
        l2_prod["pressure"].attrs[
            "source"
        ] = "computed from CAMS and DEM (see DEM metadata)"

        param = "to3c"
        l2_prod[param] = dict_attrs[param]
        l2_prod[param].attrs["unit"] = ""
        l2_prod[param].attrs["description"] = "Total columnar ozone concentration"
        l2_prod[param].attrs["source"] = "CAMS"

        param = "tno2c"
        l2_prod[param] = dict_attrs[param]
        l2_prod[param].attrs["unit"] = ""
        l2_prod[param].attrs[
            "description"
        ] = "Total columnar Nitrogen dioxide concentration"
        l2_prod[param].attrs["source"] = "CAMS"

        # -----------------------------
        # --metadata
        l2_prod.attrs = self.raster.attrs
        l2_prod.attrs["processing_date"] = str(dt.datetime.now())
        l2_prod.attrs["acquisition_date"] = str(l2_prod.attrs["acquisition_date"])
        l2_prod.attrs["hgrs_version"] = hgrs.__version__
        l2_prod.attrs["description"] = "PRISMA L2A-hGRS cube data"
        l2_prod.attrs["DEM"] = "not available"
        l2_prod.attrs["aerosol_model"] = aerosol_model

        for key in keys:
            l2_prod.attrs[key] = str(dict_attrs[key])
