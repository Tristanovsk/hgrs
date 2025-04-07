# python

import os, copy
import glob

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr

import datetime as dt
import logging

import hgrs.driver as driver
import hgrs

opj = os.path.join

HGRSDATA = '/DATA/git/satellite_app/hgrs/data/lut'
TOALUT = 'opac_osoaa_lut_v3.nc'
TRANSLUT = 'transmittance_downward_irradiance.nc'


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
                l1c_path,
                l2c_path,
                cams_path
                ):

        # ---------------------------------------
        # Load pre-computed radiative transfer LUT
        # ---------------------------------------
        logging.info('Load pre-computed radiative transfer LUT')
        aero_lut = xr.open_dataset(self.lut_file)
        aero_lut['wl'] = aero_lut['wl'] * 1000

        Ttot_Ed = xr.open_dataset(self.trans_lut_file)

        # ---------------------------------------
        # construct L1C image plus angle rasters
        # ---------------------------------------
        logging.info('construct L1C image plus angle rasters')
        try:
            dc_l1c = driver.read_L1C_data(l1c_path, reflectance_unit=True, drop_vars=True)
            dc_l2c = driver.read_L2C_data(l2c_path)
        except:
            logging.info('input file format not recognized, stop')
            return

        for param in ['sza', 'vza', 'raa']:
            dc_l1c[param] = dc_l2c[param]
        del dc_l2c

        # get L1C object
        self.l1_prod = dc_l1c

        date = dc_l1c.attrs['acquisition_date']
        clon = np.nanmean(dc_l1c.lon)
        clat = np.nanmean(dc_l1c.lat)

        # -----------------------------------------
        # Create hGRS object
        # -----------------------------------------
        logging.info('Create hGRS object')
        prod = hgrs.algo(dc_l1c, xcoarsen=10, ycoarsen=10)
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

        rh = '_rh70'
        models = ['COAV' + rh, 'COPO' + rh, 'DESE' + rh, 'MACL' + rh, 'MAPO' + rh,
                  ]  # 'ANTA' + rh, 'ARCT' + rh,'URBA' + rh
        lut_aod = aero_lut.aot.sel(model=models, aot_ref=1).interp(wl=cams_aod.wl)
        idx = np.abs((cams_aod / cams.aod550) - lut_aod).sum('wl').argmin()
        opac_model = aero_lut.sel(model=models).model.values[idx]

        # set gases and pressure
        prod.pressure = float(cams.sp) * 1e-2
        prod.to3c = float(cams.gtco3)
        prod.tno2c = float(cams.tcno2)
        prod.tch4c = float(cams.tc_ch4)

        # -----------------------------------------
        # Apply water masking
        # -----------------------------------------
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
        wv_retrieval = hgrs.water_vapor(prod)
        wv_retrieval.solve()
        prod.get_wv_transmittance_raster(wv_retrieval.water_vapor)
        prod.water_vapor_correction()

        # ------------------------------------------
        # aerosol retrieval
        # ------------------------------------------
        logging.info('aerosol retrieval')

        variable = 'Rtoa_masked'
        prod.coarse_masked_raster = prod.remove_wl_dataset(
            prod.coarse_masked_raster, prod.wl_to_remove, variable=variable)

        # TODO double check regularization from CAMS AOT values
        aod550_mean = cams.aod550.mean().values

        aod550_std = cams.aod550.std().values
        aod550_std = np.max([aod550_std, 0.2 * aod550_mean+0.05])
        aot550_min = np.max([aod550_mean - 2*aod550_std,0.001])
        aero_retrieval = hgrs.aerosol(prod,
                                      aerosol_model=opac_model,
                                      first_guess=[aod550_mean, 0.],
                                      aot550_limits=[aot550_min,
                                                     aod550_mean + aod550_std])
        aero_retrieval.solve()
        aero_retrieval.get_atmo_parameters(prod.coarse_masked_raster.wl)

        # ------------------------------------------
        # full resolution processing
        # ------------------------------------------
        logging.info('process full resolution')

        prod.raster = prod.remove_wl_dataset(prod.raster, prod.wl_to_remove)
        prod.other_gas_correction(raster_name='raster', variable='Rtoa_masked')
        wv_full = prod.get_full_resolution(wv_retrieval.water_vapor)
        prod.get_wv_transmittance_raster(wv_full)
        prod.water_vapor_correction(raster_name='raster', variable='Rtoa_masked')

        Rdiff_full = aero_retrieval.atmo_img.Rtoa_diff  # .interp(x=prod.raster.x, y=prod.raster.y)
        Tdir_full = aero_retrieval.atmo_img.Tdir  # .interp(x=prod.raster.x, y=prod.raster.y)
        Rcorr = (prod.raster.Rtoa_masked - Rdiff_full)
        wl_sunglint = prod.wl_sunglint
        sunglint_eps = aero_retrieval.sunglint_eps
        BRDF_sunglint = (Rcorr.sel(wl=wl_sunglint) / (Tdir_full.sel(wl=wl_sunglint)
                                                      * sunglint_eps.sel(wl=wl_sunglint))).mean(dim='wl')
        BRDF_sunglint.name = 'brdfg_full'
        BRDF_sunglint = BRDF_sunglint.reset_coords().brdfg_full
        # TODO clean up xarray inheritance of some extra coordinates...
        # BRDF_sunglint = BRDF_sunglint.drop_vars('aot_ref', errors=False).squeeze()
        Rdir = Tdir_full * sunglint_eps * BRDF_sunglint

        Rrs_l2 = (Rcorr - Rdir) / np.pi

        # finally correct for down and upward transmittances
        # TODO compute pixel wise
        aot_ref = float(aero_retrieval.aero_img.aot_ref.mean())
        wl = Rrs_l2.wl.values
        sza = float(Rrs_l2.sza)
        vza = float(Rrs_l2.vza)
        Ttot_Ed_ = Ttot_Ed.Ttot_Ed.sel(model=opac_model).interp(sza=sza, method='cubic'
                                                                ).interp(aot_ref=aot_ref, method='quadratic').interp(
            wl=wl, method='cubic')
        Ttot_Lu_ = Ttot_Ed.Ttot_Ed.sel(model=opac_model).interp(sza=vza, method='cubic'
                                                                ).interp(aot_ref=aot_ref, method='quadratic').interp(
            wl=wl, method='cubic') ** 1.05
        Ttot = (Ttot_Ed_ * Ttot_Lu_).reset_coords(drop=True)
        Rrs_l2 = Rrs_l2 / Ttot
        Rrs_l2.name = 'Rrs'

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
        geom = prod.raster[['lon', 'lat']].drop_vars('tcwv')
        Rrs_ = Rrs_l2.reset_coords().drop_vars(['model', 'z']).rename({'tcwv': 'tcwv_full', 'aot_ref': 'aot_ref_full'})
        l2_prod = xr.merge([Rrs_, wv, aero, geom, water_pixel_prop])
        l2_prod['brdfg_full'] = BRDF_sunglint

        l2_prod.lat.attrs['unit'] = 'degrees_north'
        l2_prod.lat.attrs['long_name'] = 'latitude'

        l2_prod.lon.attrs['unit'] = 'degrees_east'
        l2_prod.lon.attrs['long_name'] = 'longitude'

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
        param = 'aot_ref_full'
        l2_prod[param].attrs['unit'] = '-'
        l2_prod[param].attrs['long_name'] = 'aerosol_optical_thickness'
        l2_prod[param].attrs['description'] = 'Aerosol optical thickness at the reference wavelength (550nm)'

        param = 'tcwv'
        l2_prod[param].attrs['unit'] = 'kg m-2'
        l2_prod[param].attrs['long_name'] = 'total_columnar_water_vapor'
        l2_prod[param].attrs['description'] = 'Water vapor integrated over the atmospheric layer'
        param = 'tcwv_std'
        l2_prod[param].attrs['unit'] = 'kg m-2'
        l2_prod[param].attrs['long_name'] = 'total_columnar_water_vapor_standard_deviation'
        l2_prod[param].attrs['description'] = 'Uncertainty based on optimal estimation procedure'
        param = 'tcwv_full'
        l2_prod[param].attrs['unit'] = 'kg m-2'
        l2_prod[param].attrs['long_name'] = 'total_columnar_water_vapor'
        l2_prod[param].attrs['description'] = 'Water vapor integrated over the atmospheric layer'

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
        l2_prod.attrs['hgrs_version'] = hgrs.__version__
        l2_prod.attrs['description'] = 'PRISMA L2A-hGRS cube data'
        l2_prod.attrs['DEM'] = 'not available'
        l2_prod.attrs['aerosol_model'] = aero_retrieval.aerosol_model
        keys = ['wl_water_vapor', 'wl_sunglint', 'wl_atmo', 'wl_to_remove', 'wl_green', 'wl_nir', 'wl_1600', 'wl_rgb',
                'xcoarsen', 'ycoarsen', 'Npix_per_megapix', 'block_size', 'pixel_percentage', 'pixel_threshold',
                'ang_resol', 'dirdata', 'abs_gas_file', 'lut_file', 'water_vapor_transmittance_file',
                'sunglint_threshold',
                'ndwi_threshold', 'green_swir_index_threshold', 'pressure', 'to3c', 'tno2c', 'tch4c', 'psl',
                'coef_abs_scat',
                'altitude']
        for key in keys:
            l2_prod.attrs[key] = str(prod.__dict__[key])

        self.l2_prod =l2_prod
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
            'aot_ref_full': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True,
                             "complevel": complevel},
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
            'tcwv_full': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True,
                          "complevel": complevel},
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
        #l2_prod.close()
        return
