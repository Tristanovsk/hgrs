# python

import os, copy

import glob

from tqdm.auto import tqdm

import numpy as np
import scipy.optimize as so
import xarray as xr

import datetime as dt
import logging

import hgrs.driver as driver
import hgrs
from hgrs.config import behavior

opj = os.path.join

class Process():
    def __init__(self):
        self.successful = False

    def execute(self,
                img_path,
                cams_path,
                *,
                sensor='enmap',
                geoproject=True,
                sun_azimuth=None,
                sun_elevation=None,
                satellite_inclination=None,
                look_angle=None,
                view_azimuth=None,
                view_zenith=None,
                ortho=False,
                ):

        # ---------------------------------------
        # construct L1C image plus angle rasters
        # ---------------------------------------
        logging.info('construct L1C image plus angle rasters')
        # action = 'load L1C image plus angle rasters'
        # pbar = tqdm(total=len(action),
        #             desc=action + f": {img_path} ")
        if sensor == 'emit':
            logging.info('Opening EMIT L1B radiance image')
            geometry = {
                'sun_azimuth': sun_azimuth,
                'sun_elevation': sun_elevation,
                'view_azimuth': view_azimuth,
                'view_zenith': view_zenith,
            }
            missing = [key for key, value in geometry.items() if value is None]
            if missing:
                raise ValueError(
                    'EMIT processing requires scene geometry parameters: '
                    + ', '.join(missing)
                )
            try:
                driver = hgrs.Driver('emit')
                l1_prod = driver.read_emit(
                    img_path, **geometry, geoproject=geoproject, ortho=ortho,
                )
            except Exception as e:
                logging.exception('Could not read EMIT input %s', img_path)
                raise RuntimeError('EMIT input could not be read') from e
        elif sensor == 'hyperion':
            logging.info('Opening Hyperion L1R image')
            geometry = {
                'sun_azimuth': sun_azimuth,
                'sun_elevation': sun_elevation,
                'satellite_inclination': satellite_inclination,
                'look_angle': look_angle,
            }
            missing = [key for key, value in geometry.items() if value is None]
            if missing:
                raise ValueError(
                    'Hyperion processing requires scene geometry parameters: '
                    + ', '.join(missing)
                )
            try:
                driver = hgrs.HyperionDriver(img_path)
                l1_prod = driver.read(**geometry, geoproject=geoproject)
            except Exception as e:
                logging.exception('Could not read Hyperion input %s', img_path)
                raise RuntimeError('Hyperion input could not be read') from e
        elif sensor == 'tanager':
            logging.info('Opening Tanager image')
            try:
                driver = hgrs.Driver("tanager")
                l1_prod = driver.read_tanager(
                    img_path,
                    reflectance_unit=True,
                    geoproject=geoproject,
                )
            except Exception as e:
                logging.exception('Could not read Tanager input %s', img_path)
                raise RuntimeError('Tanager input could not be read') from e
        elif isinstance(img_path, str):
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
                                             drop_vars=True,
                                             geoproject=geoproject)
            except Exception as e:
                logging.info('input file format not recognized, stop')
                raise RuntimeError from e

        # get L1C object
        self.l1_prod = l1_prod

        acquisition_time = driver.scene_metadata.acquisition_time
        if isinstance(acquisition_time, xr.DataArray):
            acquisition_time = acquisition_time.values
        if isinstance(acquisition_time, np.ndarray):
            if acquisition_time.size != 1:
                raise ValueError('Scene acquisition time must be scalar')
            acquisition_time = acquisition_time.reshape(-1)[0]
        # CAMS timestamps are UTC and timezone-naive. Normalize Python
        # timezone-aware datetimes to naive UTC, then use NumPy datetime64 for
        # selection against xarray's decoded time coordinate.
        if isinstance(acquisition_time, dt.datetime):
            if acquisition_time.tzinfo is not None:
                acquisition_time = acquisition_time.astimezone(
                    dt.timezone.utc
                ).replace(tzinfo=None)
            date = np.datetime64(acquisition_time, 'us')
        else:
            date = np.datetime64(acquisition_time)
        if geoproject:
            raster = l1_prod.sza.rio.reproject(4326)
            clon, clat = float(raster.x.mean()), float(raster.y.mean())
        else:
            clon = float(l1_prod.lon.mean())
            clat = float(l1_prod.lat.mean())
        # pbar.refresh()

        # -----------------------------------------
        # Load CAMS data for this scene
        # -----------------------------------------
        logging.info('get CAMS data for scene')
        # Reanalysis files use ``valid_time`` while forecast files use ``time``
        # (or forecast_period/forecast_reference_time before normalization).
        # Do not request chunks for a dimension that may not exist.
        cams = xr.open_dataset(cams_path, decode_cf=True)

        # fix for new ADS format (sept 2024)
        if ('forecast_period' in cams.dims) & ('forecast_reference_time' in cams.dims):
            cams = cams.stack(time_buffer=['forecast_period', 'forecast_reference_time']).swap_dims(
                {'time_buffer': 'valid_time'}).sortby('valid_time').rename(
                {'valid_time': 'time'}).drop_vars(['time_buffer'])
        elif 'time' not in cams.dims and 'valid_time' in cams.dims: # reanalysis
            cams = cams.rename({'valid_time': 'time'})

        # Match the selection scalar to the decoded CAMS coordinate dtype.
        # Use the same datetime unit as decoded CAMS files where supported.
        cams_time_dtype = cams.time.dtype
        if np.issubdtype(cams_time_dtype, np.datetime64):
            date = date.astype(cams_time_dtype)
        cams = cams.sel(time=date, method='nearest')
        cams = cams.sel(latitude=clat, longitude=clon, method='nearest')

        # -----------------------------------------
        # Create hGRS object
        # -----------------------------------------
        logging.info('Create hGRS object')
        prod = hgrs.Algo(
            l1_prod,
            cams,
            scene_metadata=driver.scene_metadata,
            sensor_description=driver.sensor_description,
        )
        prod.round_angles()

        # -----------------------------------------
        # Apply cloud, water masking
        # -----------------------------------------

        # TODO put omnimask settings (bands) in default_config.yml
        logging.info('Apply omnicloudmask')
        red_index = 670
        green_index = 550
        nir_index = 940
        rgnir = prod.raster.Rtoa.sel(
            wl=[red_index, green_index, nir_index], method='nearest'
        ).fillna(0)
        omnimask = prod.get_omnicloudmask(rgnir)
        # OmniCloudMask classes 1 and 2 are thick/thin cloud; class 3 is
        # cloud shadow and is intentionally not labeled as cloud here.
        prod.raster['cloud_mask'] = (omnimask.isin([1, 2])).rename('cloud_mask')
        prod.raster['cloud_mask'].attrs.update(
            long_name='cloud mask',
            description='True for OmniCloudMask thick or thin cloud pixels; False otherwise',
        )
        logging.info('Apply water masking')
        prod.apply_water_masks()
        # Evaluate spectral water tests on the original Rtoa values so a cloud
        # classification does not change the independent water-mask meaning.
        prod.raster['Rtoa'] = prod.raster['Rtoa'].where(omnimask == 0)
        # Keep the non-land-masked radiance for full-resolution correction;
        # the land mask is only used to select pixels for retrievals.
        if driver.sensor_description.name == "PRISMA":
            full_resolution_rtoa = prod.raster.Rtoa.copy(deep=True)
            prod.apply_land_mask()

        # -----------------------------------------
        # Construct coarse resolution raster
        # -----------------------------------------
        logging.info('Construct coarse resolution raster')
        prod.get_coarse_masked_raster()
        if driver.sensor_description.name == "PRISMA":

            prod.raster['Rtoa'] = full_resolution_rtoa
            del full_resolution_rtoa
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

        aerosol_state = prod.aerosol_parameters
        self.aerosol_parameters = aerosol_state
        aero_retrieval = hgrs.Aerosol(prod)
        aero_retrieval.solve()
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
        logging.info('Begin water vapor correction')
        chunk = 256
        height, width, Nwl = len(prod.raster.y), len(prod.raster.x), len(prod.raster.wl)
        variable = 'Rtoa'
        logging.info(
            'Full-resolution water-vapor correction starts: '
            'Rtoa shape=%s, Twv shape=%s, chunk=%d',
            prod.raster[variable].shape,
            prod.Twv_raster.shape,
            chunk,
        )
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
        wl_sunglint = prod.water_parameters.wl_sunglint
        logging.info('Begin aerosol correction and sunglint removal')

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
        # Preserve native PRISMA geolocation in the L2A file when processing
        # without reprojection, so the product can be projected later.
        if not geoproject:
            missing_geolocation = {
                name for name in ('lon', 'lat') if name not in prod.raster
            }
            if missing_geolocation:
                raise ValueError(
                    'Native-geometry output requires source geolocation arrays; '
                    f'missing {sorted(missing_geolocation)}'
                )
            l2_prod['lon'] = prod.raster.lon
            l2_prod['lat'] = prod.raster.lat
            l2_prod['lon'].attrs.update(long_name='longitude', units='degrees_east')
            l2_prod['lat'].attrs.update(long_name='latitude', units='degrees_north')

        logging.info('Final transmittance correction')

        # Apply down/up transmittance using the configured spatial scope.
        aot_ref = float(aero_retrieval.aero_img.aot_ref.mean())
        wl = l2_prod.Rrs.wl.values
        sza = float(aero_retrieval.sza)
        vza = float(aero_retrieval.vza)
        aerosol_model = aerosol_state.aerosol_model
        Ttot_lut = prod.lut_tables.Ttot_Ed.Ttot_Ed.sel(model=aerosol_model)
        if prod.atmospheric_correction.bidirectional_transmittance_global:
            Ttot_Ed_ = Ttot_lut.interp(sza=sza, method='cubic').interp(
                aot_ref=aot_ref, method='quadratic'
            ).interp(wl=wl, method='cubic')
            Ttot_Lu_ = Ttot_lut.interp(sza=vza, method='cubic').interp(
                aot_ref=aot_ref, method='quadratic'
            ).interp(wl=wl, method='cubic') ** 1.05
            Ttot = (Ttot_Ed_ * Ttot_Lu_).reset_coords(drop=True)
        else:
            assert not behavior.PREVIOUS_BEHAVIOR
            aot_field = aero_retrieval.aero_img.aot_ref_smoothed
            valid_aot = aot_field.notnull()
            aot_values = np.unique(aot_field.values[valid_aot.values])
            if aot_values.size == 0:
                Ttot = xr.full_like(l2_prod.Rrs, np.nan)
            else:
                # Evaluate each distinct AOT once; LUT interpolation with a
                # spatially varying xarray target creates colliding dimensions.
                aot_da = xr.DataArray(aot_values, dims='aot_ref', coords={'aot_ref': aot_values})
                Ttot_Ed_ = Ttot_lut.interp(sza=sza, method='cubic').interp(
                    aot_ref=aot_da, method='quadratic'
                ).interp(wl=wl, method="cubic")
                Ttot_Lu_ = Ttot_lut.interp(sza=vza, method='cubic').interp(
                    aot_ref=aot_da, method='quadratic'
                ).interp(wl=wl, method="cubic")

                Ttot_by_aot = Ttot_Ed_ * Ttot_Lu_**1.05
                Ttot_by_aot = Ttot_by_aot.assign_coords(aot_ref=aot_values)
                Ttot = Ttot_by_aot.sel(aot_ref=aot_field.round(3), method='nearest')
                Ttot = Ttot.drop_vars('aot_ref', errors='ignore').where(valid_aot)
                Ttot = Ttot.interp(x=prod.raster.x, y=prod.raster.y)
        l2_prod['Rrs'] = l2_prod.Rrs / Ttot

        # -----------------------------
        # construct output image
        # -----------------------------
        logging.info('construct final product')

        # -----------------------------
        # data
        wv = wv_retrieval.water_vapor.rename({"x": "xc", "y": "yc"})
        aero = aero_retrieval.aero_img.rename({"x": "xc", "y": "yc"})
        water_pixel_number = prod.coarse_masked_raster.water_pixel_number
        if 'tcwv' in water_pixel_number.coords:
            water_pixel_number = water_pixel_number.drop_vars('tcwv')
        water_pixel_prop = (
            water_pixel_number / prod.Npix_per_megapix
        ).rename({"x": "xc", "y": "yc"})
        water_pixel_prop.name = 'water_pix_prop'
        # Rrs can retain the TCWV lookup target as an auxiliary coordinate.
        # The retrieved coarse-grid TCWV is merged below as a data variable.
        if 'tcwv' in l2_prod.coords:
            l2_prod = l2_prod.drop_vars('tcwv')
        # geom = prod.raster[['lon', 'lat']].drop_vars('tcwv')
        # Rrs_ = Rrs_l2.reset_coords().drop_vars(['model', 'z']).rename({'tcwv': 'tcwv_full', 'aot_ref': 'aot_ref_full'}).set_coords(['time','spatial_ref'])
        l2_prod = xr.merge([l2_prod, wv, aero, water_pixel_prop])
        for mask_name in ('water_mask', 'cloud_mask'):
            l2_prod[mask_name] = prod.raster[mask_name].astype(bool)
        if "landcover_mask" in prod.raster:
            l2_prod["landcover_mask"] = prod.raster.landcover_mask
        if hasattr(prod, "land_mask"):
            l2_prod["land_mask"] = prod.land_mask
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

        product_parameters = prod.return_dictionary()
        param = 'to3c'
        l2_prod[param] = product_parameters[param]
        l2_prod[param].attrs['unit'] = ''
        l2_prod[param].attrs['description'] = 'Total columnar ozone concentration'
        l2_prod[param].attrs['source'] = 'CAMS'

        param = 'tno2c'
        l2_prod[param] = product_parameters[param]
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
        l2_prod.attrs['geoprojection'] = 'native' if not geoproject else 'reprojected'
        for key, value in product_parameters.items():
            l2_prod.attrs[key] = str(value)

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

        for mask_name in (
            'water_mask',
            'cloud_mask',
            'water_validity_mask',
            'aerosol_validity_mask',
            'water_vapor_validity_mask',
            'aerosol_retrieval_validity_mask',
        ):
            if mask_name in self.l2_prod:
                encoding[mask_name] = {
                    'dtype': 'uint8', 'zlib': True, 'complevel': complevel
                }

        # clean up before exporting netcdf output
        if os.path.exists(ofile):
            os.remove(ofile)

        odir = os.path.dirname(ofile)
        if not os.path.exists(odir):
            os.mkdir(odir)

        output_product = self.l2_prod.sel(wl=slice(400, 1150))

        # Keep NetCDF raster chunks small and CF coordinate variables clean
        # for the newly added sensor products. The default xarray chunking
        # splits Rrs across many spectral bands, so reading one QGIS band can
        # require decompressing a very large 3D chunk.
        sensor = output_product.attrs.get('platform')
        source_name = str(output_product.attrs.get('L1C_product_name', '')).lower()
        if sensor not in {'Tanager', 'Hyperion', 'EMIT'}:
            if 'basic_radiance_hdf5' in source_name:
                sensor = 'Tanager'
            elif source_name.endswith('.l1r'):
                sensor = 'Hyperion'
        if sensor in {'Tanager', 'Hyperion', 'EMIT'}:
            height = output_product.sizes['y']
            width = output_product.sizes['x']
            encoding['Rrs']['chunksizes'] = (
                1, min(256, height), min(256, width)
            )
            if 'brdfg_full' in output_product:
                encoding['brdfg_full'] = {
                    'dtype': 'int16', 'scale_factor': 0.00001,
                    'add_offset': .2, '_FillValue': -32768, 'zlib': True,
                    'complevel': complevel,
                    'chunksizes': (min(256, height), min(256, width)),
                }


            # Re-encoding an opened NetCDF dataset can otherwise preserve its
            # old auxiliary-coordinate list, including the scalar variables
            # removed above. Let xarray rebuild coordinates from the current
            # dataset structure.
            for variable in output_product.variables.values():
                variable.encoding.pop('coordinates', None)
            # CF coordinate variables should not use a NaN _FillValue.
            for coordinate in ('x', 'y', 'wl', 'xc', 'yc'):
                if coordinate in output_product.coords:
                    encoding[coordinate] = {'_FillValue': None}

        # NetCDF attributes do not support boolean values. Correction helpers
        # attach boolean flags to variables, and xarray carries those attrs
        # into the final product. Store them as the equivalent 0/1 byte.
        def netcdf_safe_attrs(attrs):
            return {
                key: np.int8(value) if isinstance(value, (bool, np.bool_)) else value
                for key, value in attrs.items()
            }

        output_product.attrs = netcdf_safe_attrs(output_product.attrs)
        for variable in output_product.variables.values():
            variable.attrs = netcdf_safe_attrs(variable.attrs)

        output_product.to_netcdf(ofile, encoding=encoding)
        # l2_prod.close()
        return
