import os
import yaml
import xarray as xr
import numpy as np
from hgrs.config import get_config

opj = os.path.join

config = get_config()

HGRSDATA = config["path"]["data_root"]
TOALUT = config["path"]["toa_lut"]
TRANSLUT = config["path"]["trans_lut"]


class LUTTables:
    """
    Attributes:
        - gaz_lut(xr.Dataset): ch4, co, co2, h2o, n2o, o2, o3, o4 (optical thickness) (@1000hPa) -> dim wl=21687
        - aero_lut(xr.Dataset): 12 variables in function of wl, aot_ref, sza, vza, azi, model (aerosol). -> dim wl=10
        - Ttot_Ed:(xr.DataArray): normalized downwelling irradiance in function of wl, aot_ref, sza, wind, model (aerosol) -> dim wl=10
        - Twv_lu(xr.DataArray): LUT for water vapor transmittance in function of wl, air_mass, total columnar water vapor)
    """

    # get LUT
    def __init__(self, sensor_description):
        # pre-computed auxiliary data

        lut_file = opj(HGRSDATA, TOALUT)
        trans_lut_file = opj(HGRSDATA, TRANSLUT)
        abs_gas_file = opj(HGRSDATA, "lut_abs_opt_thickness_normalized.nc")
        self.abs_gas_file = abs_gas_file
        self.lut_file = lut_file
        self.trans_lut_file = trans_lut_file

        self.gas_lut = xr.open_dataset(abs_gas_file)
        self.aero_lut = xr.open_dataset(lut_file).isel(wind=1)
        self.Ttot_Ed = xr.open_dataset(trans_lut_file).isel(wind=1).Ttot_Ed

        air_masses = np.array(
            [
                *np.linspace(2, 6, 41),
                6.5,
                7.0,
                7.5,
                8.0,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                20,
                30,
            ]
        )
        air_masses = xr.DataArray(air_masses, coords={"air_mass": air_masses})
        tcwvs = np.array(
            [0, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 60]
        )
        tcwvs = xr.DataArray(tcwvs, coords={"tcwv": tcwvs})
        ot_wv = self.gas_lut.h2o * tcwvs
        Twv_lut = np.exp(-air_masses * ot_wv)
        self.Twv_lut = sensor_description.sensor_mod.convolve(Twv_lut)

        # convert wavelength in nanometer
        self.aero_lut["wl"] = self.aero_lut["wl"] * 1000
        self.aero_lut["wl"].attrs[
            "description"
        ] = "wavelength of simulation (nanometer)"
        self.Ttot_Ed["wl"] = self.Ttot_Ed["wl"] * 1e3
        self.Ttot_Ed["wl"].attrs["description"] = "wavelength of simulation (nanometer)"
