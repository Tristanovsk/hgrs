"""Load and prepare the lookup tables used during atmospheric correction."""

import numpy as np
import xarray as xr

from .config import behavior

from importlib.resources import files

WATER_VAPOR_TRANSMITTANCE_FILE = str(files(__package__)/'data/lut/water_vapor_transmittance.nc')

class LUTTables:
    """Radiative-transfer tables prepared for one sensor grid."""

    def __init__(
        self,
        *,
        lut_file,
        trans_lut_file,
        abs_gas_file,
    ):
        self.lut_file = str(lut_file)
        self.trans_lut_file = str(trans_lut_file)
        self.abs_gas_file = str(abs_gas_file)
        self.water_vapor_transmittance_file = WATER_VAPOR_TRANSMITTANCE_FILE

        self.gas_lut = xr.open_dataset(self.abs_gas_file)
        self.aero_lut = xr.open_dataset(self.lut_file).isel(wind=1)
        self.Ttot_Ed = xr.open_dataset(self.trans_lut_file).isel(wind=1)

        self.aero_lut["wl"] = self.aero_lut["wl"] * 1000
        self.aero_lut["wl"].attrs["description"] = (
            "wavelength of simulation (nanometer)"
        )
        self.Ttot_Ed["wl"] = self.Ttot_Ed["wl"] * 1e3
        self.Ttot_Ed["wl"].attrs["description"] = (
            "wavelength of simulation (nanometer)"
        )

        if behavior.PREVIOUS_BEHAVIOR:
            # Preserve the rework's original precomputed water-vapor table.
            self.full_Twv_lut = xr.open_dataset(WATER_VAPOR_TRANSMITTANCE_FILE)
        else:
            self.full_Twv_lut = self._build_twv_lut() # 

    def _build_twv_lut(self):
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
        tcwv_values = np.array(
            [0, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 60]
        )
        tcwvs = xr.DataArray(tcwv_values, coords={"tcwv": tcwv_values})
        optical_thickness = self.gas_lut.h2o * tcwvs
        transmittance = np.exp(-air_masses * optical_thickness)
        return transmittance.rename("Twv").to_dataset() # convolve with sensor response in interp_twv()

        
    def interp_twv(self, wavelengths, sensor_description):
        """Interpolate Twv to requested sensor bands.

        The sensor convolution can produce the complete acquisition grid while
        a caller asks for only a subset. Interpolation selects that subset on
        its requested ``wl`` coordinate.
        """
        if isinstance(wavelengths, xr.DataArray) and wavelengths.dims == ("wl",):
            target_wavelengths = wavelengths
        else:
            target_values = np.asarray(wavelengths)
            target_wavelengths = xr.DataArray(
                target_values,
                dims=("wl",),
                coords={"wl": target_values},
            )
        if behavior.PREVIOUS_BEHAVIOR:
            return self.full_Twv_lut.interp(wl=target_wavelengths)

        transmittance = self.full_Twv_lut.Twv
        transmittance = sensor_description.convolve(transmittance)
        wavelength_dimension = (
            "wl_sensor" if "wl_sensor" in transmittance.dims else "wl"
        )
        transmittance = transmittance.sel(
            {wavelength_dimension: target_wavelengths}
        )
        if wavelength_dimension != "wl" and wavelength_dimension in transmittance.coords:
            transmittance = transmittance.drop_vars(wavelength_dimension)
        return transmittance.rename("Twv").to_dataset()
    