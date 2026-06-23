# python

import os, copy

import importlib_resources
import yaml
import glob

from tqdm.auto import tqdm

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr

import datetime as dt
import logging

import hgrs.driver as driver
import hgrs

opj = os.path.join

configfile = importlib_resources.files(__package__).joinpath('config.yml')
with open(configfile, 'r') as file:
    config = yaml.safe_load(file)

HGRSDATA = config['path']['data_root']
TOALUT = config['path']['toa_lut']
TRANSLUT = config['path']['trans_lut']

LUT_FILE = opj(HGRSDATA, TOALUT)
TRANS_LUT_FILE = opj(HGRSDATA, TRANSLUT)


class Process():
    def __init__(self):
        self.lut_file = opj(HGRSDATA, TOALUT)
        self.trans_lut_file = opj(HGRSDATA, TRANSLUT)
        # self.cams_dir = CAMS_PATH
        # self.Nproc = NCPU
        self.pressure_ref = 101500.
        self.flags_tokeep = [3]
        self.flags_tomask = [0, 1, 10, 13, 14, 18]
        self.successful = False

    def execute(self,
                img_path,
                cams_path
                ):

        # ---------------------------------------
        # construct L1C image plus angle rasters
        # ---------------------------------------
        logging.info('construct L1C image plus angle rasters')
        # action = 'load L1C image plus angle rasters'
        # pbar = tqdm(total=len(action),
        #             desc=action + f": {img_path} ")
        if isinstance(img_path, str):
            logging.info('Opening EnMAP image')

            try:
                driver = hgrs.Driver('enmap')
                l1_prod = driver.read_l1c_enmap(img_path, reflectance_unit=True)
            except:
                logging.info('input file format not recognized, stop')
                return
        else:
            logging.info('Opening PRISMA image')
            print(img_path)
            try:
                driver = hgrs.Driver('prisma')

                l1_prod = driver.read_prisma(img_path[0],
                                             img_path[1],
                                             reflectance_unit=True,
                                             drop_vars=True)
            except:
                logging.info('input file format not recognized, stop')
                return

        # get L1C object
        self.l1_prod = l1_prod

        date = l1_prod.time
        raster = l1_prod.sza.rio.reproject(4326)
        clon, clat = float(raster.x.mean()), float(raster.y.mean())
        # pbar.refresh()

        # -----------------------------------------
        # Create hGRS object
        # -----------------------------------------
        logging.info('Create hGRS object')
        prod = hgrs.Algo(l1_prod, xcoarsen=20, ycoarsen=20)
        prod.round_angles()

        # -----------------------------------------
        # get CAMS and set atmospheric parameters
        # -----------------------------------------
        logging.info('get CAMS and set atmospheric parameters')
        # lazy loading
        cams = xr.open_dataset(cams_path, decode_cf=True,
                               chunks={'time': 1, 'x': 500, 'y': 500})

        # fix for new ADS format (sept 2024)
        if ('forecast_period' in cams.dims) & ('forecast_reference_time' in cams.dims):
            cams = cams.stack(time_buffer=['forecast_period', 'forecast_reference_time']).swap_dims(
                {'time_buffer': 'valid_time'}).sortby('valid_time').rename(
                {'valid_time': 'time'}).drop_vars(['time_buffer'])

        # slicing
        cams = cams.sel(time=date, method='nearest')
        cams = cams.sel(latitude=clat, longitude=clon, method='nearest')

        # select OPAC aerosol model
        # aod = cams[['aod355', 'aod380', 'aod400', 'aod440', 'aod469', 'aod500', 'aod550', 'aod645', 'aod670',
        #            'aod800', 'aod865', 'aod1020', 'aod1064', 'aod1240', 'aod1640', 'aod2130']].to_pandas()
        # aod.index = aod.index.str.replace('aod', '').astype(int)
        # cams_aod = aod.to_xarray().rename({'index': 'wl'})
        cams_wls = [469, 550, 670, 865, 1240]
        param_aod = []
        for wl in cams_wls:
            wl_ = str(wl)
            param_aod.append('aod' + wl_)

        cams_aod = cams[param_aod].to_array(dim='wl')

        wl_cams = cams_aod.wl.str.replace('aod', '').astype(float)
        cams_aod = cams_aod.assign_coords(wl=wl_cams)

        # new LUT:
        lut_aod = prod.aero_lut.aot.sel(aot_ref=1).interp(wl=cams_aod.wl)
        idx = np.abs((cams_aod / cams.aod550) - lut_aod).sum('wl').argmin()
        opac_model = prod.aero_lut.model.values[idx]
        logging.info('OPAC model: ' + opac_model)

        # set gases and pressure
        prod.pressure = float(cams.sp) * 1e-2
        prod.to3c = float(cams.gtco3)
        prod.tno2c = float(cams.tcno2)
        prod.tch4c = float(cams.tc_ch4)

        # -----------------------------------------
        # Apply cloud, water masking
        # -----------------------------------------

        # TODO put omnimask settins (bands) in config.yml
        logging.info('Apply omnicloudmask')
        red_index = 670
        green_index = 550
        nir_index = 940
        rgnir = prod.raster.Rtoa.sel(wl=[red_index, green_index, nir_index], method='nearest').fillna(0)  # .values
        omnimask = prod.get_omnicloudmask(rgnir)
        prod.raster['Rtoa'] = prod.raster['Rtoa'].where(omnimask == 0)

        logging.info('Apply water masking')
        prod.apply_water_masks()

        # -----------------------------------------
        # Construct coarse resolution raster
        # -----------------------------------------
        logging.info('Construct coarse resolution raster')
        prod.get_coarse_masked_raster()
        # prod.plot_water_pix_number()

        # -----------------------------------------
        # Correct for gaseous absorption
        # -----------------------------------------
        logging.info('Correct for gaseous absorption')
        prod.get_gaseous_transmittance()
        prod.other_gas_correction()

        # ------------------------------------------
        # water vapor retrieval and correction
        # ------------------------------------------
        logging.info('water vapor retrieval and correction')
        wv_retrieval = hgrs.WaterVapor(prod)
        wv_retrieval.solve()
        prod.get_wv_transmittance_raster(wv_retrieval.water_vapor)
        prod.water_vapor_correction()
        logging.info('mask bands where gaseous abs. is too strong')
        Tg_tot = prod.Tg_other * prod.Twv_raster.mean(['x', 'y'])

        # ------------------------------------------
        # aerosol retrieval
        # ------------------------------------------
        logging.info('aerosol retrieval')

        variable = 'Rtoa'
        # prod.coarse_masked_raster = prod.remove_wl_dataset(
        #    prod.coarse_masked_raster, prod.wl_to_remove, variable=variable)
        prod.coarse_masked_raster = prod.remove_wl_dataset(
            prod.coarse_masked_raster, prod.wl_to_remove, variable=variable)

        # remove bands where Tg is below a threshold (typically Tg < 0.5)
        prod.coarse_masked_raster[variable] = prod.coarse_masked_raster[variable].where(Tg_tot > 0.5, drop=True)
        prod.raster[variable] = prod.raster[variable].where(Tg_tot > 0.5, drop=True)

        # TODO double check regularization from CAMS AOT values
        aod550_mean = cams.aod550.mean().values

        aod550_std = cams.aod550.std().values
        aod550_std = np.max([aod550_std, 0.2 * aod550_mean + 0.05])
        aot550_min = 0.002  # np.max([aod550_mean - 2*aod550_std,0.001])
        aero_retrieval = hgrs.Aerosol(prod,
                                      aerosol_model=opac_model,
                                      first_guess=[aod550_mean, 0.],
                                      aot550_limits=[aot550_min,
                                                     aod550_mean + 2 * aod550_std])
        aero_retrieval.solve()
        aero_retrieval.prepare_lut(prod.coarse_masked_raster.wl)
        weights = prod.coarse_masked_raster['water_pixel_number'].__deepcopy__().to_numpy().astype(float)
        aero_retrieval.smoothing(weights)

        # self.aero_img['aot_ref_smoothed']=self.aero_img['aot_ref_smoothed']*0.5
        # construct aot raster
        # with full res
        # aot_ref_vals = self.aot_ref_full.round(3)
        # with coarse res

        aot_ref_vals = aero_retrieval.aero_img['aot_ref_smoothed'].round(3)

        aot_refs = np.unique(aot_ref_vals)
        aot_refs = aot_refs[~np.isnan(aot_refs)]
        # TODO update LUT for aot< 0.001
        aot_refs[aot_refs < 0.002] = 0.002
        # if rounded aot_ref has unique value
        if len(aot_refs) == 1:
            aot_refs = np.concatenate([aot_refs, 1.2 * aot_refs])
        print(aot_refs)
        aots = aero_retrieval.aot_lut.interp(aot_ref=aot_refs, method='linear')
        aots = aots.interp(aot_ref=aot_ref_vals, method='nearest')

        aots.name = 'aot'
        aots.attrs['description'] = 'spectral aerosol optical thickness'

        # construct raster for diffuse atmospheric reflectance
        # TODO check quadratic interpolation (should be much better)
        Rdiffs = aero_retrieval.Rtoa_lut.interp(aot_ref=aot_refs, method='linear')
        Rdiffs = Rdiffs.interp(aot_ref=aot_ref_vals, method='nearest')
        Rdiffs.name = 'Rtoa_diff'
        Rdiffs.attrs['description'] = 'top-of-atmosphere atmosphere reflectance'

        # construct raster for direct transmittance due to rayleigh and aerosol
        Tdirs = aero_retrieval.transmittance_dir(aots, aero_retrieval.air_mass, rot=aero_retrieval.rot)
        Tdirs.name = 'Tdir'
        Tdirs.attrs['description'] = 'direct transmittance due to rayleigh and aerosol for total air mass'

        aero_retrieval.get_atmo_parameters(prod)
        self.aero_retrieval = aero_retrieval

        # ------------------------------------------
        # full resolution processing
        # ------------------------------------------
        logging.info('process full resolution')

        prod.raster = prod.remove_wl_dataset(prod.raster, prod.wl_to_remove)
        prod.other_gas_correction(raster_name='raster', variable='Rtoa')

        # ------------------------------------------
        # water vapor
        # ------------------------------------------
        chunk = 256
        height, width, Nwl = len(prod.raster.y), len(prod.raster.x), len(prod.raster.wl)
        results = np.full((height, width), 0, dtype=np.float32)
        variable = 'Rtoa'
        for iy in range(0, height, chunk):
            yc = min(height, iy + chunk)
            if yc > height:
                continue
            for ix in range(0, width, chunk):
                xc = min(width, ix + chunk)
                if xc > width:
                    continue
                raster = prod.raster[variable][:, iy:yc, ix:xc]
                Twv_raster = prod.Twv_raster.interp(x=raster.x, y=raster.y)
                prod.raster[variable].data[:, iy:yc, ix:xc] = raster / Twv_raster

        Rdiff_full = aero_retrieval.atmo_img.Rtoa_diff  # .interp(x=prod.raster.x, y=prod.raster.y)
        Tdir_full = aero_retrieval.atmo_img.Tdir  # .interp(x=prod.raster.x, y=prod.raster.y)
        wl_sunglint = prod.wl_sunglint

        Rrs = np.full((Nwl, height, width), np.nan, dtype=np.float32)
        BRDF_sunglint = np.full((height, width), np.nan, dtype=np.float32)

        for iy in range(0, height, chunk):
            yc = min(height, iy + chunk)
            if yc > height:
                continue
            for ix in range(0, width, chunk):
                xc = min(width, ix + chunk)
                if xc > width:
                    continue

                Rcorr = prod.raster.Rtoa[:, iy:yc, ix:xc]
                Rdiff_full_ = Rdiff_full.interp(x=Rcorr.x, y=Rcorr.y)

                Rcorr = Rcorr - Rdiff_full_

                Tdir_full_ = Tdir_full.interp(x=Rcorr.x, y=Rcorr.y)

                sunglint_eps = aero_retrieval.sunglint_eps
                BRDF_sunglint[iy:yc, ix:xc] = (Rcorr.sel(wl=wl_sunglint) / (Tdir_full_.sel(wl=wl_sunglint)
                                                                            * sunglint_eps.sel(wl=wl_sunglint))).mean(
                    dim='wl')

                # TODO clean up xarray inheritance of some extra coordinates...
                # BRDF_sunglint = BRDF_sunglint.drop_vars('aot_ref', errors=False).squeeze()
                Rdir = Tdir_full_ * sunglint_eps * BRDF_sunglint[iy:yc, ix:xc]

                Rrs[:, iy:yc, ix:xc] = (Rcorr - Rdir) / np.pi

        l2_prod = xr.Dataset(dict(Rrs=(["wl", "y", "x"], Rrs),
                                  brdfg_full=(["y", "x"], BRDF_sunglint), ),
                             coords=dict(x=prod.raster.x,
                                         y=prod.raster.y,
                                         wl=prod.raster.wl),
                             )

        # finally correct for down and upward transmittances
        # TODO compute pixel wise
        Ttot_Ed = xr.open_dataset(TRANS_LUT_FILE).isel(wind=1)
        Ttot_Ed['wl'] = Ttot_Ed['wl'] * 1e3

        aot_ref = float(aero_retrieval.aero_img.aot_ref.mean())
        wl = l2_prod.Rrs.wl.values
        sza = float(aero_retrieval.sza)
        vza = float(aero_retrieval.vza)
        Ttot_Ed_ = Ttot_Ed.Ttot_Ed.sel(model=opac_model).interp(sza=sza, method='cubic'
                                                                ).interp(aot_ref=aot_ref, method='quadratic').interp(
            wl=wl, method='cubic')
        Ttot_Lu_ = Ttot_Ed.Ttot_Ed.sel(model=opac_model).interp(sza=vza, method='cubic'
                                                                ).interp(aot_ref=aot_ref, method='quadratic').interp(
            wl=wl, method='cubic') ** 1.05
        Ttot = (Ttot_Ed_ * Ttot_Lu_).reset_coords(drop=True)
        l2_prod['Rrs'] = l2_prod.Rrs / Ttot

        # -----------------------------
        # construct output image
        # -----------------------------
        logging.info('construct final product')

        # -----------------------------
        # data
        wv = wv_retrieval.water_vapor.rename({"x": "xc", "y": "yc"})
        aero = aero_retrieval.aero_img.rename({"x": "xc", "y": "yc"})
        water_pixel_prop = (prod.coarse_masked_raster.water_pixel_number / prod.Npix_per_megapix).drop_vars(
            'tcwv').rename({"x": "xc", "y": "yc"})
        water_pixel_prop.name = 'water_pix_prop'
        # geom = prod.raster[['lon', 'lat']].drop_vars('tcwv')
        # Rrs_ = Rrs_l2.reset_coords().drop_vars(['model', 'z']).rename({'tcwv': 'tcwv_full', 'aot_ref': 'aot_ref_full'}).set_coords(['time','spatial_ref'])
        l2_prod = xr.merge([l2_prod, wv, aero, water_pixel_prop])
        # l2_prod['brdfg_full'] = BRDF_sunglint

        param = 'Rrs'
        l2_prod[param].attrs['unit'] = 'per steradian'
        l2_prod[param].attrs['long_name'] = 'Remote sensing reflectance'
        l2_prod[param].attrs['description'] = 'Directional water-leaving radiance normalized ' + \
                                              'by downwelling irradiance in the observation geometry'

        param = 'water_pix_prop'
        l2_prod[param].attrs['unit'] = '-'
        l2_prod[param].attrs['description'] = 'Relative number of water pixel within mega-pixel used for inversion'

        param = 'brdfg'
        l2_prod[param].attrs['unit'] = '-'
        l2_prod[param].attrs['long_name'] = 'BRDF_sunglint'
        l2_prod[param].attrs['description'] = 'Bidirectional reflectance distribution function ' + \
                                              'estimated from the sunglint in the SWIR for the observation geometry'
        param = 'brdfg_std'
        l2_prod[param].attrs['unit'] = '-'
        l2_prod[param].attrs['long_name'] = 'BRDF_sunglint_standard deviation'
        l2_prod[param].attrs['description'] = 'Uncertainty based on optimal estimation procedure'
        param = 'brdfg_full'
        l2_prod[param].attrs['unit'] = '-'
        l2_prod[param].attrs['long_name'] = 'BRDF_sunglint'
        l2_prod[param].attrs['description'] = 'Bidirectional reflectance distribution function ' + \
                                              'estimated from the sunglint in the SWIR for the observation geometry'

        param = 'aot_ref'
        l2_prod[param].attrs['unit'] = '-'
        l2_prod[param].attrs['long_name'] = 'aerosol_optical_thickness'
        l2_prod[param].attrs['description'] = 'Aerosol optical thickness at the reference wavelength (550nm)'
        param = 'aot_ref_std'
        l2_prod[param].attrs['unit'] = '-'
        l2_prod[param].attrs['long_name'] = 'aerosol_optical_thickness_standard_deviation'
        l2_prod[param].attrs['description'] = 'Uncertainty based on optimal estimation procedure'
        # param = 'aot_ref_full'
        # l2_prod[param].attrs['unit'] = '-'
        # l2_prod[param].attrs['long_name'] = 'aerosol_optical_thickness'
        # l2_prod[param].attrs['description'] = 'Aerosol optical thickness at the reference wavelength (550nm)'

        param = 'tcwv'
        l2_prod[param].attrs['unit'] = 'kg m-2'
        l2_prod[param].attrs['long_name'] = 'total_columnar_water_vapor'
        l2_prod[param].attrs['description'] = 'Water vapor integrated over the atmospheric layer'
        param = 'tcwv_std'
        l2_prod[param].attrs['unit'] = 'kg m-2'
        l2_prod[param].attrs['long_name'] = 'total_columnar_water_vapor_standard_deviation'
        l2_prod[param].attrs['description'] = 'Uncertainty based on optimal estimation procedure'
        # param = 'tcwv_full'
        # l2_prod[param].attrs['unit'] = 'kg m-2'
        # l2_prod[param].attrs['long_name'] = 'total_columnar_water_vapor'
        # l2_prod[param].attrs['description'] = 'Water vapor integrated over the atmospheric layer'

        l2_prod['pressure'] = prod.pressure
        l2_prod['pressure'].attrs['unit'] = 'hPa'
        l2_prod['pressure'].attrs['description'] = 'Atmospheric pressure at the surface level'
        l2_prod['pressure'].attrs['source'] = 'computed from CAMS and DEM (see DEM metadata)'

        param = 'to3c'
        l2_prod[param] = prod.__dict__[param]
        l2_prod[param].attrs['unit'] = ''
        l2_prod[param].attrs['description'] = 'Total columnar ozone concentration'
        l2_prod[param].attrs['source'] = 'CAMS'

        param = 'tno2c'
        l2_prod[param] = prod.__dict__[param]
        l2_prod[param].attrs['unit'] = ''
        l2_prod[param].attrs['description'] = 'Total columnar Nitrogen dioxide concentration'
        l2_prod[param].attrs['source'] = 'CAMS'

        # -----------------------------
        # --metadata
        l2_prod.attrs = prod.raster.attrs
        l2_prod.attrs['processing_date'] = str(dt.datetime.now())
        l2_prod.attrs['acquisition_date'] = str(l2_prod.attrs['acquisition_date'])
        l2_prod.attrs['hgrs_version'] = hgrs.__version__
        l2_prod.attrs['description'] = 'PRISMA L2A-hGRS cube data'
        l2_prod.attrs['DEM'] = 'not available'
        l2_prod.attrs['aerosol_model'] = aero_retrieval.aerosol_model
        keys = ['wl_water_vapor', 'wl_sunglint', 'wl_atmo', 'wl_to_remove', 'wl_non_neg', 'wl_green', 'wl_nir',
                'wl_1600', 'wl_rgb',
                'xcoarsen', 'ycoarsen', 'Npix_per_megapix', 'block_size', 'pixel_percentage', 'pixel_threshold',
                'ang_resol', 'abs_gas_file', 'lut_file', 'water_vapor_transmittance_file',
                'sunglint_threshold',
                'ndwi_threshold', 'green_swir_index_threshold', 'pressure', 'to3c', 'tno2c', 'tch4c', 'psl',
                'coef_abs_scat',
                'altitude']
        for key in keys:
            print(key)
            l2_prod.attrs[key] = str(prod.__dict__[key])

        self.l2_prod = l2_prod
        self.successful = True

        return

    def write_output(self,
                     ofile):
        ######################################
        # Write final product
        ######################################
        logging.info('export final product into netcdf')
        complevel = 5
        encoding = {
            'Rrs': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True,
                    "complevel": complevel},
            # 'aot_ref_full': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
            #                 "complevel": complevel},
            'aot_ref': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
                        "complevel": complevel},
            'aot_ref_std': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
                            "complevel": complevel},
            'brdfg_full': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768,
                           "zlib": True, "complevel": complevel},
            'brdfg': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True,
                      "complevel": complevel},
            'brdfg_std': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768,
                          "zlib": True, "complevel": complevel},
            # 'tcwv_full': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True,
            #              "complevel": complevel},
            'tcwv': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True, "complevel": complevel},
            'tcwv_std': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True,
                         "complevel": complevel}}

        # clean up before exporting netcdf output
        if os.path.exists(ofile):
            os.remove(ofile)

        odir = os.path.dirname(ofile)
        if not os.path.exists(odir):
            os.mkdir(odir)

        self.l2_prod.sel(wl=slice(400, 1150)).to_netcdf(ofile, encoding=encoding)
        # l2_prod.close()
        return
