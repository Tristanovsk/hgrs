import numpy as np
import xarray as xr


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
        # TODO: altitude impact ?

        self.to3c = float(cams.gtco3)
        self.tno2c = float(cams.tcno2)
        self.tch4c = float(cams.tc_ch4)
        self.aod550 = cams.aod550
        self.aerosol_model = aerosol_model
