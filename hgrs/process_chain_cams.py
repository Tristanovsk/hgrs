import os, copy
import glob

import matplotlib as mpl

mpl.use('tkagg')
import matplotlib.pyplot as plt
import colorcet as cc

import numpy as np
import scipy.optimize as so
import pandas as pd
import xarray as xr

import datetime as dt

import hgrs.driver as driver
import hgrs

opj = os.path.join

workdir_ = '/sat_data/satellite/acix-iii/AERONET-OC'
sites = ['galataplatform', 'sanmarcoplatform', 'zeebrugge', 'lisco', 'lakeerie', 'casablanca', 'irbelighthouse',
         'ariaketower', 'kemigawa', 'uscseaprism', 'section7', 'southgreenbay', 'palgrunden', 'venezia', 'lucinda',
         'bahiablanca', 'socheongcho', 'wavecissite', 'lakeokeechobee', 'gustavdalentower']

workdir_ = '/sat_data/satellite/acix-iii'
sites = ['Wendtorf', 'Varese', 'Geneve', 'Venice_Lagoon', 'Garda', 'Trasimeno']

cams_dir = '/DATA/projet/magellium/acix-iii/cams'


for site in sites:

    workdir = opj(workdir_, site)
    odir = opj('/data/acix-iii', site)  # /sat_data/satellite/acix-iii/results/',site)
    if not os.path.exists(odir):
        os.mkdir(odir)

    for l1c_path in glob.glob(opj(workdir, '*L1_STD_OFFL*.he5')):

        l1c = os.path.basename(l1c_path)
        cams_file = l1c.replace('.he5', '_cams.nc')
        cams_path = opj(cams_dir,cams_file)

        ofile = opj(odir, l1c.replace('.he5', '.nc').replace('L1_STD_OFFL', 'L2A_hGRS_gueymard'))
        if os.path.exists(ofile):
            continue

        print(l1c)
        l2c = l1c.replace('L1_STD_OFFL', 'L2C_STD')
        l2c_path = opj(workdir, l2c)

        # ---------------------------------------
        # construct L1C image plus angle rasters
        # ---------------------------------------
        try:
            dc_l1c = driver.read_L1C_data(l1c_path, reflectance_unit=True, drop_vars=True)
            dc_l2c = driver.read_L2C_data(l2c_path)
        except:
            continue

        for param in ['sza', 'vza', 'raa']:
            dc_l1c[param] = dc_l2c[param]
        del dc_l2c

        # -----------------------------------------
        # Create hGRS object
        # -----------------------------------------
        prod = hgrs.algo(dc_l1c, xcoarsen=10, ycoarsen=10)
        prod.round_angles()

        # -----------------------------------------
        # get CAMS and set atmospheric parameters
        # -----------------------------------------
        cams = xr.open_dataset(cams_path)
        prod.pressure = float(cams.sp) * 1e-2
        prod.to3c = float(cams.gtco3)
        prod.tno2c = float(cams.tcno2)
        prod.tch4c = float(cams.tc_ch4)

        # -----------------------------------------
        # Apply water masking
        # -----------------------------------------
        prod.apply_water_masks()

        # -----------------------------------------
        # Construct coarse resolution raster
        # -----------------------------------------
        prod.get_coarse_masked_raster()
        #prod.plot_water_pix_number()

        # -----------------------------------------
        # Correct for gaseous absorption
        # -----------------------------------------
        prod.get_gaseous_transmittance()
        prod.other_gas_correction()

        # ------------------------------------------
        # water vapor retrieval and correction
        # ------------------------------------------
        wv_retrieval = hgrs.water_vapor(prod)
        wv_retrieval.solve()
        prod.get_wv_transmittance_raster(wv_retrieval.water_vapor)
        prod.water_vapor_correction()

        # ------------------------------------------
        # aerosol retrieval
        # ------------------------------------------
        variable = 'Rtoa_masked'
        prod.coarse_masked_raster = prod.remove_wl_dataset(
            prod.coarse_masked_raster, prod.wl_to_remove, variable=variable)
        aero_retrieval = hgrs.aerosol(prod)
        aero_retrieval.solve()
        aero_retrieval.get_atmo_parameters(prod.coarse_masked_raster.wl)

        # ------------------------------------------
        # full resolution processing
        # ------------------------------------------
        prod.raster = prod.remove_wl_dataset(prod.raster, prod.wl_to_remove)
        prod.other_gas_correction(raster_name='raster', variable='Rtoa_masked')
        wv_full = prod.get_full_resolution(wv_retrieval.water_vapor)
        prod.get_wv_transmittance_raster(wv_full)
        prod.water_vapor_correction(raster_name='raster', variable='Rtoa_masked')

        Rdiff_full = aero_retrieval.atmo_img.Rtoa_diff#.interp(x=prod.raster.x, y=prod.raster.y)
        Tdir_full = aero_retrieval.atmo_img.Tdir#.interp(x=prod.raster.x, y=prod.raster.y)
        Rcorr = (prod.raster.Rtoa_masked - Rdiff_full)
        wl_sunglint = prod.wl_sunglint
        sunglint_eps = aero_retrieval.sunglint_eps
        BRDF_sunglint = (Rcorr.sel(wl=wl_sunglint) / (Tdir_full.sel(wl=wl_sunglint)
                                                      * sunglint_eps.sel(wl=wl_sunglint))).mean(dim='wl')
        BRDF_sunglint.name = 'brdfg_full'
        BRDF_sunglint = BRDF_sunglint.reset_coords().brdfg_full
        # TODO clean up xarray inheritance of some extra coordinates...
        #BRDF_sunglint = BRDF_sunglint.drop_vars('aot_ref', errors=False).squeeze()
        Rdir = Tdir_full * sunglint_eps * BRDF_sunglint

        Rrs_l2 = (Rcorr - Rdir) / np.pi
        Rrs_l2.name = 'Rrs'

        # -----------------------------
        # construct output image
        # -----------------------------

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
        complevel=5
        encoding = {'Rrs': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True, "complevel": complevel},
                    'aot_ref_full': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True, "complevel": complevel},
                    'aot_ref': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True, "complevel": complevel},
                    'aot_ref_std': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -9999, "zlib": True, "complevel": complevel},
                    'brdfg_full': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True, "complevel": complevel},
                    'brdfg': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True, "complevel": complevel},
                    'brdfg_std': {'dtype': 'int16', 'scale_factor': 0.00001, 'add_offset': .2, '_FillValue': -32768, "zlib": True, "complevel": complevel},
                    'tcwv_full': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True, "complevel": complevel},
                    'tcwv': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True, "complevel": complevel},
                    'tcwv_std': {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999, "zlib": True, "complevel": complevel}}



        l2_prod.sel(wl=slice(400, 1150)).to_netcdf(ofile, encoding=encoding)
        l2_prod.close()

