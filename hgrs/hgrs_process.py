# python

import os, copy

import yaml
import glob

from tqdm.auto import tqdm

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr

import datetime as dt
import logging

import hgrs
from hgrs.hgrs_kernel import LUTTables, CAMSProduct


class Process:
    def __init__(self):

        # self.cams_dir = CAMS_PATH
        # self.Nproc = NCPU
        self.pressure_ref = 101500.0
        self.flags_tokeep = [3]
        self.flags_tomask = [0, 1, 10, 13, 14, 18]
        self.successful = False

    def execute(self, img_path, cams_path, geoproject=True):

        # ---------------------------------------
        # construct L1C image plus angle rasters
        # ---------------------------------------
        logging.info("construct L1C image plus angle rasters")
        # action = 'load L1C image plus angle rasters'
        # pbar = tqdm(total=len(action), desc=action + f": {img_path} ")
        # from pathlib import Path
        # import pickle

        # path_pkl = Path("test.pkl")
        # if not path_pkl.exists():

        if isinstance(img_path, str):
            # try:
            driver = hgrs.Driver("enmap")
            l1_prod, sensor_desc = driver.driver(img_path, reflectance_unit=True)
            # except:
            #     logging.info("input file format not recognized, stop")
            #     return
        else:
            # try:
            driver = hgrs.Driver("prisma")
            l1_prod, sensor_desc = driver.read_prisma(
                img_path[0], img_path[1], geoproject=geoproject
            )

            # except Exception as e:
            #     logging.info("input file format not recognized, stop")
            #     print(e)
            #     return
        #     with open(path_pkl, "wb") as f:
        #         pickle.dump(l1_prod, f)
        # else:
        #     with open(path_pkl, "rb") as f:
        #         l1_prod = pickle.load(f)
        # get L1C object
        self.l1_prod = l1_prod

        # -----------------------------------------
        # get CAMS and set atmospheric parameters
        # -----------------------------------------
        date = l1_prod.time

        if "UL" in l1_prod.attrs.keys():
            points = np.array(
                [
                    l1_prod.attrs["UL"],
                    l1_prod.attrs["UR"],
                    l1_prod.attrs["LL"],
                    l1_prod.attrs["LR"],
                ],
                dtype=np.float32,
            )
            clon, clat = np.mean(points[:, 0]), np.mean(points[:, 1])
        elif hasattr(l1_prod, "sza") and hasattr(l1_prod.sza.rio, "x"):

            raster = l1_prod.sza.rio.reproject(4326)
            clon, clat = float(raster.x.mean()), float(raster.y.mean())
        else:
            raise Exception(f"unable to check geoerferencing of {l1_prod}")
        # pbar.refresh()

        logging.info("get CAMS and set atmospheric parameters")
        lut_tables = LUTTables(sensor_desc)
        # lazy loading
        cams_product = CAMSProduct(cams_path, date, clat, clon, lut_tables.aero_lut)

        logging.info("OPAC model: " + cams_product.aerosol_model)
        logging.info("Load pre-computed radiative transfer LUT")

        # # set gases and pressure

        # -----------------------------------------
        # Create hGRS object
        # -----------------------------------------
        logging.info("Create hGRS object")
        prod = hgrs.Algo(
            l1_prod, cams_product, lut_tables, sensor_desc, xcoarsen=20, ycoarsen=20
        )
        self.prod = prod

        prod.round_angles()

        # -----------------------------------------
        # Apply cloud, water masking
        # -----------------------------------------
        auxdata = prod.auxdata
        auxdata
        # TODO (tristan) put omnimask settins (bands) in config.yml
        logging.info("Apply omnicloudmask")
        red_index = 670
        green_index = 550
        nir_index = 940
        rgnir = (
            prod.raster.Rtoa.sel(
                wl_sensor=[red_index, green_index, nir_index], method="nearest"
            )
            .fillna(0)
            .transpose("wl_sensor", "y", "x")
        )  # .values
        omnimask = prod.get_omnicloudmask(rgnir)
        # fill na the clouds & non water pixels
        prod.raster["Rtoa"] = prod.raster["Rtoa"].where(omnimask == 0)

        logging.info("Apply water masking")
        prod.apply_water_masks()

        # -----------------------------------------
        # Construct coarse resolution raster
        # -----------------------------------------
        logging.info("Construct coarse resolution raster")
        raster_name_coarse = "coarse_masked_raster"
        variable = "Rtoa"
        prod.get_coarse_masked_raster()
        # prod.plot_water_pix_number()

        # -----------------------------------------
        # Correct for gaseous absorption
        # -----------------------------------------
        logging.info(f"Correct {variable} for gaseous absorption")

        prod.other_gas_correction(
            raster_name=raster_name_coarse, variable=variable
        )  # state coarse (variable)

        # ------------------------------------------
        # water vapor retrieval and correction
        # ------------------------------------------
        logging.info("water vapor retrieval and correction")

        prod.find_coarse_water_vapor(raster_name=raster_name_coarse, variable=variable)
        prod.coarse_water_vapor_correction(
            raster_name=raster_name_coarse, variable=variable
        )
        # data
        logging.info(f"mask bands where gaseous abs. is too strong")
        prod.drop_band_predefined(
            raster_name_coarse, variable=variable
        )  # state coarse (bands)
        prod.drop_band_high_gaz_absorption(
            raster_name_coarse, variable=variable
        )  # state coarses (bands)

        # raster and coarse raster now have lower wl res
        # ------------------------------------------
        # aerosol retrieval
        # ------------------------------------------
        logging.info("aerosol retrieval")

        prod.find_coarse_aeroglint()

        aero_retrieval = prod.aero_retrieval

        prod.compute_atmo_img_smooth(prod.coarse_masked_raster.wl_sensor)
        self.aero_retrieval = aero_retrieval

        # ------------------------------------------
        # full resolution processing
        # ------------------------------------------
        logging.info("process full resolution")
        # ------------------------------------------
        # full resolution pre-processing
        # ------------------------------------------

        prod.drop_band_predefined("raster", variable=variable)  # state coarse (bands)
        prod.drop_band_high_gaz_absorption("raster", variable=variable)  # state (bands)

        # ------------------------------------------
        # Other Gas
        # ------------------------------------------

        prod.other_gas_correction(raster_name="raster", variable=variable)

        # ------------------------------------------
        # water vapor
        # ------------------------------------------
        prod.fullres_water_vapor_correction()

        # ------------------------------------------
        # Glint
        # ------------------------------------------

        Rrs, BRDF_sunglint = prod.fullres_aeroglint_correction()

        # ------------------------------------------
        # bidirectional transmitance (aerosols)
        # ------------------------------------------

        l2_prod = xr.Dataset(
            dict(
                Rrs=(["wl_sensor", "y", "x"], Rrs),
                brdfg_full=(["y", "x"], BRDF_sunglint),
            ),
            coords=dict(
                x=prod.raster.x,
                y=prod.raster.y,
                wl_sensor=prod.raster.wl_sensor,
                fwhm=prod.raster.fwhm,
            ),
        )

        # finally correct for down and upward transmittances (aerosols)

        Ttot = prod.compute_atmo_bidir_transmittance()

        l2_prod["Rrs"] = l2_prod.Rrs / Ttot

        # -----------------------------
        # construct output image
        # -----------------------------
        logging.info("construct final product")

        # -----------------------------
        # data
        wv = prod.wv_retrieval.water_vapor  # half shift
        aero = aero_retrieval.aero_img
        water_pixel_prop = prod.derive_water_pixel_prop()
        # geom = prod.raster[['lon', 'lat']].drop_vars('tcwv')
        # Rrs_ = Rrs_l2.reset_coords().drop_vars(['model', 'z']).rename({'tcwv': 'tcwv_full', 'aot_ref': 'aot_ref_full'}).set_coords(['time','spatial_ref'])
        l2_prod = xr.merge([l2_prod, wv, aero, water_pixel_prop])
        # l2_prod['brdfg_full'] = BRDF_sunglint
        prod.complete_L2A_attributes(l2_prod)
        self.l2_prod = l2_prod
        self.successful = True
        return

    def write_output(self, ofile):
        ######################################
        # Write final product
        ######################################
        logging.info("export final product into netcdf")
        complevel = 5
        encoding = {
            "Rrs": {
                "dtype": "int16",
                "scale_factor": 0.00001,
                "add_offset": 0.2,
                "_FillValue": -32768,
                "zlib": True,
                "complevel": complevel,
            },
            #'aot_ref_full': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
            #                 "complevel": complevel},
            "aot_ref": {
                "dtype": "int16",
                "scale_factor": 0.001,
                "_FillValue": -9999,
                "zlib": True,
                "complevel": complevel,
            },
            "aot_ref_std": {
                "dtype": "int16",
                "scale_factor": 0.001,
                "_FillValue": -9999,
                "zlib": True,
                "complevel": complevel,
            },
            "brdfg_full": {
                "dtype": "int16",
                "scale_factor": 0.00001,
                "add_offset": 0.2,
                "_FillValue": -32768,
                "zlib": True,
                "complevel": complevel,
            },
            "brdfg": {
                "dtype": "int16",
                "scale_factor": 0.00001,
                "add_offset": 0.2,
                "_FillValue": -32768,
                "zlib": True,
                "complevel": complevel,
            },
            "brdfg_std": {
                "dtype": "int16",
                "scale_factor": 0.00001,
                "add_offset": 0.2,
                "_FillValue": -32768,
                "zlib": True,
                "complevel": complevel,
            },
            #'tcwv_full': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True,
            #              "complevel": complevel},
            "tcwv": {
                "dtype": "int16",
                "scale_factor": 0.01,
                "_FillValue": -9999,
                "zlib": True,
                "complevel": complevel,
            },
            "tcwv_std": {
                "dtype": "int16",
                "scale_factor": 0.01,
                "_FillValue": -9999,
                "zlib": True,
                "complevel": complevel,
            },
        }

        # clean up before exporting netcdf output
        if os.path.exists(ofile):
            os.remove(ofile)

        odir = os.path.dirname(ofile)
        if not os.path.exists(odir):
            os.mkdir(odir)

        self.l2_prod.sel(wl_sensor=slice(400, 1150)).to_netcdf(ofile, encoding=encoding)
        # l2_prod.close()
        return
