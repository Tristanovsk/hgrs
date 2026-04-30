import os
import glob

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr

import h5py
import xml.etree.ElementTree as ET

from scipy.interpolate import RegularGridInterpolator

import datetime as dt
import logging

from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import warnings


from . import Reproj
from hgrs.spectral_sensitivity import BaselineInterp, Gaussian, SuperGaussian
from hgrs.hgrs_kernel import SensorDescription
from hgrs.auxdata import SolarIrradiance, get_air_mass
from hgrs.utils import Reproj, Misc


class Driver:
    def __init__(self, satellite="enmap"):

        self.satellite = satellite

        if "prisma" in satellite:
            self.driver = self.read_prisma
        elif "enmap" in satellite:
            self.driver = self.read_l1c_enmap
        else:
            logging.info("satellite mission not recognized, stop")
            return

    def read_prisma(
        self,
        l1c_path: str,
        l2c_path: str,
        reflectance_unit=True,
        drop_vars=True,
        geoproject=True,
        parallel=False,
    ):
        logging.info("construct L1C image plus angle rasters")

        dc_l1c = self.read_l1c_prisma(l1c_path, drop_vars=drop_vars)

        # complete L1C with L2C params
        dc_l2c = self.read_l2c_prisma(l2c_path)
        georef_params = ["sza", "vza", "raa"]
        for param in georef_params:
            dc_l1c[param] = dc_l2c[param]
        for param in ["UL", "UR", "LL", "LR"]:
            dc_l1c.attrs[param] = dc_l2c.attrs[param]
            # sza_mean = np.nanmean(l1_prod.sza)
        del dc_l2c

        air_mass = get_air_mass(dc_l1c.vza, dc_l1c.sza)
        dc_l1c["air_mass"] = air_mass
        dc_l1c.attrs["air_mass_mean"] = np.nanmean(air_mass)
        wl = np.array(dc_l1c.wl_sensor)
        fwhm = dc_l1c.fwhm.reset_coords(drop=True)
        sensor_desc = self.get_sensordesc(
            dc_l1c.sza, dc_l1c.vza, dc_l1c.raa, wl, fwhm, sensor="prisma"
        )

        # solar irradiance convolution to the PRISMA spectral response function and scaled
        # by the day of the year

        # dc_l1c = dc_l1c.chunk({'x': 200, 'y': 200, 'wl': 10})

        if geoproject:
            dc_l1c = self.reproject(dc_l1c, parallel=parallel)
        if reflectance_unit:
            self.perform_reflectance_corr(dc_l1c, sensor_desc)
            if drop_vars:
                dc_l1c = dc_l1c.drop_vars("Ltoa")
        return dc_l1c.transpose("wl_sensor", "y", "x"), sensor_desc

    def get_sensordesc(self, sza, vza, raa, wl_sensor, fwhm, sensor="prisma"):
        # sza_mean = np.nanmean(l1_prod.sza)
        # vza_mean = np.nanmean(l1_prod.vza)
        # raa_mean = np.nanmean(l1_prod.raa)
        # wl_sensor = np.array(l1_prod.wl_sensor)
        # fwhm = l1_prod.fwhm.reset_coords(drop=True)  # .to_dataframe()
        # air_mass = get_air_mass(l1_prod.vza, l1_prod.sza)
        # l1_prod["air_mass"] = air_mass
        # air_mass.values

        sza_mean = np.nanmean(sza)
        vza_mean = np.nanmean(vza)
        raa_mean = np.nanmean(raa)
        if sensor == "prisma":
            sensor_modelisation = Gaussian(wl_sensor, fwhm)

        elif sensor == "enmap":
            sensor_modelisation = SuperGaussian(wl_sensor, fwhm, expon=3)
        sensor_modelisation_lr = BaselineInterp(wl_sensor, inter_mod="quadratic")
        sensor_desc = SensorDescription(
            wl_sensor,
            fwhm,
            sza_mean,
            vza_mean,
            raa_mean,
            sensor_modelisation,
            sensor_modelisation_lr,
        )

        return sensor_desc

    def perform_reflectance_corr(self, data, sensor_desc):

        solar_irr = SolarIrradiance()
        F0 = solar_irr.tsis  # huillier # gueymard # kurucz

        date = data.time.values.astype("datetime64[us]").astype(dt.datetime)
        DOY = date.timetuple().tm_yday
        # get correction for Sun-Earth distance and correct solar irradiance
        D2 = Misc.earth_sun_correction(DOY)
        F0 = F0 * D2

        info = (
            {
                "description": "Convolved solar irradiance from TSIS data",
                "unit": "mW/m2/nm",
            },
        )
        F0_sensor = sensor_desc.sensor_mod.convolve(
            F0, dim="wl"
        )  # info=info, name="F0"
        F0_sensor.attrs = info

        data["F0"] = (["wl_sensor"], F0_sensor.values)
        data.F0.attrs["unit"] = "mW/m2/nm"
        data.F0.attrs["definition"] = (
            "Solar irradiance corrected for Sun-Earth distance"
        )
        data["Rtoa"] = np.pi * data.Ltoa / (data.F0 * np.cos(np.radians(data.sza)))
        return data

    def reproject(
        self,
        input_dataset,
        parallel=False,
        grid_mode="prisma_L2C",
        raise_ambiguous=True,
    ):

        if grid_mode == "hgrs":
            # Query the database for the most appropriate UTM CRS

            output_grid_size = (1200, 1200)
            grid_lons = np.linspace(
                input_dataset.lon.min().values,
                input_dataset.lon.max().values,
                output_grid_size[0],
            )
            grid_lats = np.linspace(
                input_dataset.lat.min().values,
                input_dataset.lat.max().values,
                output_grid_size[1],
            )
            new_grid = xr.Dataset(
                {"lat": (["lat"], grid_lats), "lon": (["lon"], grid_lons)}
            )

        elif grid_mode == "prisma_L2C":
            # same than above, exept the underlying georef is a rectilinear in target PRISMA EPSG (not lat/lon)
            points = np.array(
                [
                    input_dataset.UL,
                    input_dataset.UR,
                    input_dataset.LL,
                    input_dataset.LR,
                ],
                dtype=np.float32,
            )
            utm_crs_list = query_utm_crs_info(
                datum_name="WGS 84",
                area_of_interest=AreaOfInterest(
                    west_lon_degree=np.min(np.array(points)[:, 0]),
                    south_lat_degree=np.min(np.array(points)[:, 1]),
                    east_lon_degree=np.max(np.array(points)[:, 0]),
                    north_lat_degree=np.max(np.array(points)[:, 1]),
                ),
            )
            if len(utm_crs_list) > 1:
                if raise_ambiguous:
                    raise Exception("Zone intersect several crs")

                warnings.warn("Zone intersect several crs")
                utm_crs_list = query_utm_crs_info(
                    datum_name="WGS 84",
                    area_of_interest=AreaOfInterest(
                        west_lon_degree=np.mean(np.array(points)[:, 0]),
                        south_lat_degree=np.mean(np.array(points)[:, 1]),
                        east_lon_degree=np.mean(np.array(points)[:, 0]),
                        north_lat_degree=np.mean(np.array(points)[:, 1]),
                    ),
                )

            epsg = f"{utm_crs_list[0].auth_name}:{utm_crs_list[0].code}"
            transformer = Transformer.from_crs(4326, epsg, always_xy=True)
            utm_coords = np.array(
                [transformer.transform(*p) for p in points], dtype=np.float32
            )

            # 1. Find the bounds for the data
            min_e = utm_coords[:, 0].min()
            max_e = utm_coords[:, 0].max()
            min_n = utm_coords[:, 1].min()
            max_n = utm_coords[:, 1].max()

            # 2. Define the grid
            # Let's assume a 10m GSD (Ground Sample Distance)
            # If you need a specific shape (e.g. 1000x1000), use np.linspace instead
            gsd = 30.0

            grid_e = np.arange(
                min_e, max_e + max_e * np.finfo(np.float32).resolution, gsd
            )
            grid_n = np.arange(
                max_n, min_n - min_n * np.finfo(np.float32).resolution, -gsd
            )  # Decreasing for North-to-South

            ee, nn = np.meshgrid(grid_e, grid_n)
            lons, lats = transformer.transform(
                ee.flatten(), nn.flatten(), direction="INVERSE"
            )
            lons, lats = lons.reshape(ee.shape), lats.reshape(ee.shape)
            # rectilinear grid in EPSG is curvilinear in lat/lon (only degree suported in xesmf)
            new_grid = xr.Dataset(
                coords={
                    "lat": (["lat", "lon"], lats),
                    "lon": (["lat", "lon"], lons),
                }
            )
        else:
            raise NotImplementedError(f"grid_mode {grid_mode} is not implemented")
        new_grid = new_grid.chunk({"lat": 50, "lon": 50})
        output_dataset = Reproj().regridding(input_dataset, new_grid, parallel=parallel)
        output_dataset = output_dataset.rename({"lon": "x", "lat": "y"})
        output_dataset.attrs.update(input_dataset.attrs)
        if grid_mode == "prisma_L2C":
            output_dataset["x"] = grid_e
            output_dataset["y"] = grid_n
            output_dataset.rio.write_crs(epsg, inplace=True)
            output_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
            output_dataset.rio.write_coordinate_system(inplace=True)

        elif grid_mode == "hgrs":
            # put "x","y" naming:

            # adding the CRS
            d_input_crs = 4326
            output_dataset.rio.write_crs(d_input_crs, inplace=True)
            output_dataset.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
            output_dataset.rio.write_coordinate_system(inplace=True)
        else:
            raise NotImplementedError(f"grid_mode {grid_mode} is not implemented")

        return output_dataset

    def read_l1c_prisma(self, l1c_path: str, drop_vars=False):
        """
        Load PRISMA L1C data into xarray rasters
        :param l1c_path: absolute path to the .h5 prisma file
        :param reflectance_unit: to convert from TOA radiance to TOA reflectance
        :param drop_vars: if True remove the radiance raster to keep reflectance only
        :return:
        """
        # =============================================================================
        # Load geolocation, solar irradiance and TOA radiance
        # =============================================================================
        ds = h5py.File(l1c_path)

        # Geolocation
        lat = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR"][:].T
        lon = ds["/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR"][:].T
        xdim, ydim = lat.shape

        # Wavelength / fwhm
        wl = ds.attrs["List_Cw_Vnir"][5:]
        wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
        fwhm = ds.attrs["List_Fwhm_Vnir"][5:]
        fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])

        sort_index = np.argsort(wl)
        wl = wl[sort_index]
        fwhm = fwhm[sort_index]
        fwhm = xr.DataArray(
            data=fwhm,
            name="fwhm",
            coords=dict(wl_sensor=wl),
            attrs=dict(description="PRISMA relative spectral response parameter"),
        )
        date_str = ds.attrs["Product_StartTime"].decode("UTF-8")

        # DN to TOA radiance
        gain = {
            "vnir": ds.attrs["ScaleFactor_Vnir"],
            "swir": ds.attrs["ScaleFactor_Swir"],
        }
        offset = {
            "vnir": ds.attrs["Offset_Vnir"],
            "swir": ds.attrs["Offset_Swir"],
        }

        # -------------------------------------------------------------------------------
        VNIR = np.moveaxis(
            ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"][:, 5:, :]
            / gain["vnir"]
            - offset["vnir"],
            [0, 1, 2],
            [1, 2, 0],
        )  # 2 missing wls for vnir
        SWIR = np.moveaxis(
            ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][:, :-2, :]
            / gain["swir"]
            - offset["swir"],
            [0, 1, 2],
            [1, 2, 0],
        )
        Ltoa = np.dstack((VNIR, SWIR))
        del VNIR, SWIR

        Ltoa = Ltoa[:, :, sort_index]
        Ltoa = np.array(Ltoa)  # make array c-contiguous

        data = xr.Dataset(
            data_vars=dict(
                Ltoa=(["y", "x", "wl_sensor"], Ltoa),
                fwhm=(["wl_sensor"], fwhm.values),
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            coords=dict(
                x=np.arange(xdim)[::-1],
                y=np.arange(ydim)[::-1],
                time=dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f"),
                wl_sensor=wl,
            ),
            attrs=dict(description="PRISMA L1C cube data"),
        )

        # chunk data for dask
        # data = data.chunk({'x': 200, 'y': 200, 'wl': -1})

        # =============================================================================
        # Load other metadata
        # =============================================================================
        data.attrs["L1C_product_name"] = os.path.basename(l1c_path)
        data.attrs["acquisition_date"] = date_str

        # =============================================================================
        # Load masks
        # =============================================================================
        data = data.assign(
            cloud_mask=(
                ["y", "x"],
                ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/Cloud_Mask"][:].T,
            )
        )
        data = data.assign(
            sunglint_mask=(
                ["y", "x"],
                ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SunGlint_Mask"][:].T,
            )
        )
        data = data.assign(
            landcover_mask=(
                ["y", "x"],
                ds["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/LandCover_Mask"][:].T,
            )
        )

        return data.sel(wl_sensor=slice(350, 2550))

    def read_l2c_prisma(self, l2c_path: str):
        """
        Load PRISMA L2C data (including observation angles) into xarray rasters
        :param l2c_path: absolute path to the .h5 prisma file
        :return:
        """

        # =============================================================================
        # Load geolocation, solar irradiance and TOA radiance
        # =============================================================================
        ds = h5py.File(l2c_path)
        # Geolocation

        points = (
            ds.attrs["Product_ULcorner_long"],
            ds.attrs["Product_ULcorner_lat"],
            ds.attrs["Product_URcorner_long"],
            ds.attrs["Product_URcorner_lat"],
            ds.attrs["Product_LLcorner_long"],
            ds.attrs["Product_LLcorner_lat"],
            ds.attrs["Product_LRcorner_long"],
            ds.attrs["Product_LRcorner_lat"],
        )
        points = list(zip(points[::2], points[1::2]))

        lat = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Latitude"][:].T
        lon = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Longitude"][:].T
        xdim, ydim = lat.shape

        UL, UR, LL, LR = points

        # Wavelength / fwhm
        wl = ds.attrs["List_Cw_Vnir"][3:]
        wl = np.append(wl, ds.attrs["List_Cw_Swir"][:-2])
        fwhm = ds.attrs["List_Fwhm_Vnir"][3:]
        fwhm = np.append(fwhm, ds.attrs["List_Fwhm_Swir"][:-2])
        sort_index = np.argsort(wl)
        wl = wl[sort_index]
        fwhm = fwhm[sort_index]

        # # Thuillier solar irradiance convolved to the PRISMA ISRF and scaled
        # # by the day of the year
        # I0 = load_thuillier_solar_spectrum(wl, fwhm)
        # DOY = dt.datetime.strptime(ds.attrs["Product_StartTime"].decode('UTF-8'),
        #                                  "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
        # U = 1 - 0.01672 * np.cos(0.9856 * (DOY - 4))
        # I0 = I0 * U

        # DN to TOA radiance
        gain = {
            "vnir_min": ds.attrs["L2ScaleVnirMin"],
            "vnir_max": ds.attrs["L2ScaleVnirMax"],
            "swir_min": ds.attrs["L2ScaleSwirMin"],
            "swir_max": ds.attrs["L2ScaleSwirMax"],
        }

        # -------------------------------------------------------------------------------
        VNIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/VNIR_Cube"][:, 3:, :]
        VNIR = gain["vnir_min"] + VNIR * (gain["vnir_max"] - gain["vnir_min"]) / 65535
        VNIR = np.moveaxis(VNIR, [0, 1, 2], [1, 2, 0])
        SWIR = ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube"][:, :-2, :]
        SWIR = gain["swir_min"] + SWIR * (gain["swir_max"] - gain["swir_min"]) / 65535
        SWIR = np.moveaxis(SWIR, [0, 1, 2], [1, 2, 0])
        # print(f"VNIR shape = {VNIR.shape}")
        # print(f"SWIR shape = {SWIR.shape}")
        rho = np.dstack((VNIR, SWIR))
        rho = rho[:, :, sort_index]
        del VNIR, SWIR

        # # -------------------------------------------------------------------------------
        data = xr.Dataset(
            data_vars=dict(
                rho=(["y", "x", "wl_sensor"], rho),
                lon=(["y", "x"], lon),
                lat=(["y", "x"], lat),
            ),
            coords=dict(x=np.arange(xdim)[::-1], y=np.arange(ydim)[::-1], wl_sensor=wl),
            attrs=dict(description="PRISMA L2C cube data"),
        )

        # =============================================================================
        # Load other metadata
        # =============================================================================
        data.attrs["L2C_product_name"] = os.path.basename(l2c_path)
        data.attrs["acquisition_date"] = ds.attrs["Product_StartTime"].decode("UTF-8")

        # =============================================================================
        # Load geometries
        # =============================================================================
        data = data.assign(
            vza=(
                ["y", "x"],
                ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Observing_Angle"][:].T,
            )
        )
        data = data.assign(
            raa=(
                ["y", "x"],
                ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Rel_Azimuth_Angle"][
                    :
                ].T,
            )
        )
        data = data.assign(
            sza=(
                ["y", "x"],
                ds["/HDFEOS/SWATHS/PRS_L2C_HCO/Geometric Fields/Solar_Zenith_Angle"][
                    :
                ].T,
            )
        )

        # =============================================================================
        # Read atmospheric data
        # =============================================================================
        hdf_variables = ["AOT", "AEX", "WVM", "COT"]
        ds_variables = ["aot", "aex", "wvm", "cot"]
        dims = {
            "AOT": ["y2", "x2"],
            "AEX": ["y2", "x2"],
            "WVM": ["y", "x"],
            "COT": ["y", "x"],
        }
        for ii, var in enumerate(hdf_variables):
            gain_min = ds.attrs[f"L2Scale{var}Min"]
            gain_max = ds.attrs[f"L2Scale{var}Max"]
            matrix = ds[f"/HDFEOS/SWATHS/PRS_L2C_{var}/Data Fields/{var}_Map"][:].T
            var_dims = dims[var]
            data = eval(
                f"data.assign({ds_variables[ii]}=({var_dims},gain_min + matrix*(gain_max-gain_min)/65535))"
            )
        x2dim, y2dim = data["aot"].shape
        data = data.assign_coords(
            {"x2": np.arange(x2dim)[::-1], "y2": np.arange(y2dim)[::-1]}
        )

        data.attrs["UL"] = UL
        data.attrs["UR"] = UR
        data.attrs["LL"] = LL
        data.attrs["LR"] = LR
        return data

    def read_l1c_enmap(
        self,
        l1c_path: str,
        reflectance_unit=False,
        drop_vars=False,
        filter_bad_bands=True,
        expon=1.8,
    ):

        for ext in ["BIL", "TIF"]:
            l1c_raster_path = glob.glob(
                os.path.join(l1c_path, "*SPECTRAL_IMAGE." + ext)
            )
            if len(l1c_raster_path) > 0:
                l1c_raster_path = l1c_raster_path[0]
                break

        metadata_path = glob.glob(os.path.join(l1c_path, "*METADATA*.XML"))[0]

        # get XML metadata
        tree = ET.parse(metadata_path)
        root = tree.getroot()
        specific = root.find("specific")
        fwhm, offset, gain = {}, {}, {}
        for child in root.find("specific").find("bandCharacterisation"):
            wl = child.findtext("wavelengthCenterOfBand")
            fwhm[wl] = child.findtext("FWHMOfBand")
            offset[wl] = child.findtext("OffsetOfBand")
            gain[wl] = child.findtext("GainOfBand")

        fwhm = (
            pd.DataFrame(fwhm.items(), columns=["wl_sensor", "fwhm"])
            .astype(float)
            .set_index("wl_sensor")
            .to_xarray()
        )
        offset = (
            pd.DataFrame(offset.items(), columns=["wl_sensor", "offset"])
            .astype(float)
            .set_index("wl_sensor")
            .to_xarray()
        )
        gain = (
            pd.DataFrame(gain.items(), columns=["wl_sensor", "gain"])
            .astype(float)
            .set_index("wl_sensor")
            .to_xarray()
        )
        metadata = xr.merge([fwhm, offset, gain])

        # get respective indexes of vnir and swir sensors
        self.vnir_idx = (
            np.array(
                root.find("specific")
                .find("vnirProductQuality")
                .findtext("expectedChannelsList")
                .split(",")
            ).astype(int)
            - 1
        )
        self.swir_idx = (
            np.array(
                root.find("specific")
                .find("swirProductQuality")
                .findtext("expectedChannelsList")
                .split(",")
            ).astype(int)
            - 1
        )

        ### set wavelength to keep (EnMAP shows strange values over the overlap between vnir and swir sensors
        self.vnir_idx_tokeep = self.vnir_idx[:-13]
        self.swir_idx_tokeep = self.swir_idx[3:]

        date_str = specific.findtext("datatakeStart").strip().replace("Z", "")

        # open raster
        data = rxr.open_rasterio(
            l1c_raster_path,
            chunks={"x": 512, "y": 512, "band": 20},
            mask_and_scale=True,
        ).to_dataset(name="Ltoa")

        if ".BIL" in l1c_raster_path:
            data = data.swap_dims({"band": "wavelength"}).rename(
                {"wavelength": "wl_sensor"}
            )
        else:
            # get FWHM data and scale radiance
            data = data.rename({"band": "wl_sensor"})
            data["wl_sensor"] = metadata.wl_sensor
            data["fwhm"] = metadata.fwhm
            data["Ltoa"] = metadata.gain * data["Ltoa"] + metadata.offset

        data = data.transpose("wl_sensor", "y", "x")
        # data['time'] = dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        # data = data.set_coords('time')

        # convert from W.m-2.sr-1.nm-1 to mW.m-2.sr-1.nm-1
        data["Ltoa"] = 1e3 * data["Ltoa"].drop_attrs()
        data["Ltoa"].attrs["unit"] = "mW.m-2.sr-1.nm-1"
        data["Ltoa"].attrs["description"] = "top-of-atmosphere radiance"
        data["Ltoa"].attrs["name"] = "radiance"

        x = data.x.values
        y = data.y.values

        x_coarse = [x[0], x[-1]]
        y_coarse = [y[0], y[-1]]
        XX, YY = np.meshgrid(x, y)
        points = np.stack([YY.ravel(), XX.ravel()], axis=-1)

        sza_values = np.zeros((2, 2))
        sza_values[0, 0] = 90 - float(
            specific.find("sunElevationAngle").findtext("upper_left")
        )
        sza_values[0, 1] = 90 - float(
            specific.find("sunElevationAngle").findtext("upper_right")
        )
        sza_values[1, 0] = 90 - float(
            specific.find("sunElevationAngle").findtext("lower_left")
        )
        sza_values[1, 1] = 90 - float(
            specific.find("sunElevationAngle").findtext("lower_right")
        )
        sza_interp = RegularGridInterpolator(
            (y_coarse, x_coarse), sza_values, method="linear"
        )
        sza = sza_interp(points).reshape(YY.shape)

        saa_values = np.zeros((2, 2))
        saa_values[0, 0] = float(
            specific.find("sunAzimuthAngle").findtext("upper_left")
        )
        saa_values[0, 1] = float(
            specific.find("sunAzimuthAngle").findtext("upper_right")
        )
        saa_values[1, 0] = float(
            specific.find("sunAzimuthAngle").findtext("lower_left")
        )
        saa_values[1, 1] = float(
            specific.find("sunAzimuthAngle").findtext("lower_right")
        )
        saa_interp = RegularGridInterpolator(
            (y_coarse, x_coarse), saa_values, method="linear"
        )
        saa = saa_interp(points).reshape(YY.shape)

        vza_values = np.zeros((2, 2))
        vza_values[0, 0] = float(
            specific.find("viewingZenithAngle").findtext("upper_left")
        )
        vza_values[0, 1] = float(
            specific.find("viewingZenithAngle").findtext("upper_right")
        )
        vza_values[1, 0] = float(
            specific.find("viewingZenithAngle").findtext("lower_left")
        )
        vza_values[1, 1] = float(
            specific.find("viewingZenithAngle").findtext("lower_right")
        )
        vza_interp = RegularGridInterpolator(
            (y_coarse, x_coarse), vza_values, method="linear"
        )
        vza = vza_interp(points).reshape(YY.shape)

        vaa_values = np.zeros((2, 2))
        vaa_values[0, 0] = float(
            specific.find("viewingAzimuthAngle").findtext("upper_left")
        )
        vaa_values[0, 1] = float(
            specific.find("viewingAzimuthAngle").findtext("upper_right")
        )
        vaa_values[1, 0] = float(
            specific.find("viewingAzimuthAngle").findtext("lower_left")
        )
        vaa_values[1, 1] = float(
            specific.find("viewingAzimuthAngle").findtext("lower_right")
        )
        vaa_interp = RegularGridInterpolator(
            (y_coarse, x_coarse), vaa_values, method="linear"
        )
        vaa = vaa_interp(points).reshape(YY.shape)

        raa = (saa - vaa) % 360

        solar_irr = SolarIrradiance()
        F0 = solar_irr.tsis  # thuillier # gueymard # kurucz

        ## Compute irradiance for the Day Of the Year (date of acquisition)
        DOY = dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f").timetuple().tm_yday
        # get correction for Sun-Earth distance and correct solar irradiance
        D2 = Misc.earth_sun_correction(DOY)
        F0 = F0 * D2
        self.F0 = F0

        data = xr.Dataset(
            data_vars=dict(
                Ltoa=data.Ltoa,
                fwhm=(["wl_sensor"], data.fwhm.values),
                sza=(["y", "x"], sza),
                saa=(["y", "x"], saa),
                vza=(["y", "x"], vza),
                vaa=(["y", "x"], vaa),
                raa=(["y", "x"], raa),
            ),
            coords=dict(
                x=data.x.values,
                y=data.y.values,
                wl_sensor=data.wl_sensor.values,
                time=dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f"),
            ),
            attrs=dict(description="EnMAP L1C cube data"),
        )
        air_mass = get_air_mass(data.vza, data.sza)
        data["air_mass"] = air_mass
        data.attrs["air_mass_mean"] = np.nanmean(air_mass)

        wl = np.array(data.wl_sensor)
        fwhm = data.fwhm.reset_coords(drop=True)
        sensor_desc = self.get_sensordesc(
            data.sza, data.vza, data.raa, wl, fwhm, sensor="enmap"
        )
        if reflectance_unit:
            self.perform_reflectance_corr(data, sensor_desc)
            if drop_vars:
                data = data.drop_vars("Ltoa")

        # discard angles outside the image frame
        params = ["sza", "vza", "raa", "saa", "vaa"]
        for i in range(len(params)):
            data[params[i]] = data[params[i]].where(data.Rtoa.isel(wl_sensor=0) > 0)

        wl_vnir = data.wl_sensor.isel(wl_sensor=self.vnir_idx).values
        wl_swir = data.wl_sensor.isel(wl_sensor=self.swir_idx).values
        if filter_bad_bands:
            wl_vnir = data.wl_sensor.isel(wl_sensor=self.vnir_idx_tokeep).values
            wl_swir = data.wl_sensor.isel(wl_sensor=self.swir_idx_tokeep).values
            data = data.isel(wl_sensor=[*self.vnir_idx_tokeep, *self.swir_idx_tokeep])

        # Attributes

        data.attrs["L1C_product_name"] = os.path.basename(l1c_path)
        data.attrs["acquisition_date"] = dt.datetime.strptime(
            date_str, "%Y-%m-%dT%H:%M:%S.%f"
        )
        data.attrs["vnir_index"] = self.vnir_idx
        data.attrs["swir_index"] = self.swir_idx
        data.attrs["vnir_index_tokeep"] = self.vnir_idx_tokeep
        data.attrs["swir_index_tokeep"] = self.swir_idx_tokeep
        data.attrs["vnir_bands"] = wl_vnir
        data.attrs["swir_bands"] = wl_swir

        data.sza.attrs["definition"] = " Sun Zenith Angle"
        data.saa.attrs["definition"] = " Sun Azimuth Angle"
        data.vza.attrs["definition"] = " Viewing Zenith Angle"
        data.vaa.attrs["definition"] = " Viewing Azimuth Angle"
        data.raa.attrs["definition"] = " Relative Azimuth Angle"

        data.sza.attrs["unit"] = "degree"
        data.saa.attrs["unit"] = "degree"
        data.vza.attrs["unit"] = "degree"
        data.vaa.attrs["unit"] = "degree"
        data.raa.attrs["unit"] = "degree"

        data.F0.attrs["unit"] = "mW/m2/nm"
        data.Rtoa.attrs["definition"] = "Reflectance at the Top of the Atmosphere"
        data.Rtoa.attrs["unit"] = "-"
        data.F0.attrs["definition"] = (
            "Solar irradiance corrected for Sun-Earth distance"
        )
        data.Ltoa.attrs["unit"] = "mW/m2/sr/nm"
        data.Ltoa.attrs["definition"] = "Top-of-atmosphere radiance"
        data.fwhm.attrs["unit"] = "nm"
        data.fwhm.attrs["definition"] = "Full Width at Half Maximum"
        # data.attrs['crs'] = CRS.from_epsg(32630)

        return data, sensor_desc
