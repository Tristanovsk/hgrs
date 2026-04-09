import os
import importlib_resources
import yaml
import xarray as xr

opj = os.path.join

# TODO (Antoine): make it compatible with pip install
configfile = (
    importlib_resources.files(__package__).joinpath("..").joinpath("config.yml")
)
with open(configfile, "r") as file:
    config = yaml.safe_load(file)

HGRSDATA = config["path"]["data_root"]
TOALUT = config["path"]["toa_lut"]
TRANSLUT = config["path"]["trans_lut"]


class LUTTables:
    """
    Attributes:
        - gaz_lut(xr.Dataset): ch4, co, co2, h2o, n2o, o2, o3, o4 (optical thickness) (@1000hPa)
        - aero_lut(xr.Dataset): 12 variables in function of wl, aot_ref, sza, vza, azi, model (aerosol).
        - Ttot_Ed:(xr.DataArray): normalized downwelling irradiance in function of wl, aot_ref, sza, wind, model (aerosol)
        - Twv_lu(xr.DataArray): LUT for water vapor transmittance in function of wl, air_mass, total columnar water vapor)
    """

    # get LUT
    def __init__(self, sensor_description):
        # pre-computed auxiliary data

        lut_file = opj(HGRSDATA, TOALUT)
        trans_lut_file = opj(HGRSDATA, TRANSLUT)
        abs_gas_file = opj(HGRSDATA, "lut_abs_opt_thickness_normalized.nc")
        water_vapor_transmittance_file = opj(HGRSDATA, "water_vapor_transmittance.nc")
        # TODO (antoine) recompute water vapor transmittance for current sensor
        # TODO (antoine) convolutions ?
        self.abs_gas_file = abs_gas_file
        self.lut_file = lut_file
        self.trans_lut_file = trans_lut_file
        self.water_vapor_transmittance_file = water_vapor_transmittance_file

        self.gas_lut = xr.open_dataset(abs_gas_file)
        self.aero_lut = xr.open_dataset(lut_file).isel(wind=1)
        self.Ttot_Ed = xr.open_dataset(trans_lut_file).isel(wind=1).Ttot_Ed
        try:
            wl_lut = xr.open_dataset(water_vapor_transmittance_file)

            self.Twv_lut = sensor_description.sensor_mod.convolve(wl_lut.Twv)
        except KeyError:
            raise ValueError(
                f"LUT for water vapor transmitance is missing some wl, maybe it was generated the wrong captor characteristics. Wl raster/ LUT: {wl_sensor} {wl_lut.wl}"
            )
        # convert wavelength in nanometer
        self.aero_lut["wl"] = self.aero_lut["wl"] * 1000
        self.aero_lut["wl"].attrs[
            "description"
        ] = "wavelength of simulation (nanometer)"
        self.Ttot_Ed["wl"] = self.Ttot_Ed["wl"] * 1e3
        self.Ttot_Ed["wl"].attrs["description"] = "wavelength of simulation (nanometer)"
