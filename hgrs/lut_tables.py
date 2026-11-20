"""Load and prepare the lookup tables used during atmospheric correction."""

import numpy as np
import xarray as xr

from .config import behavior

from importlib.resources import files

WATER_VAPOR_TRANSMITTANCE_FILE = str(files(__package__)/'data/lut/water_vapor_transmittance.nc')

#: air masses and water vapor columns (kg m-2) of the water vapor transmittance table
TWV_AIR_MASSES = np.array([*np.linspace(2, 6, 41), 6.5, 7.0, 7.5, 8.0, 9, 10, 11, 12, 13, 14, 15, 20, 30])
TWV_TCWV = np.array([0, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 60])


def air_mass_nodes(air_mass, nodes=TWV_AIR_MASSES):
    """The two nodes of ``nodes`` around ``air_mass`` (the first or last two outside the range): the linear
    interpolation at ``air_mass`` on them is that on all the nodes"""
    j = int(np.clip(np.searchsorted(nodes, air_mass), 1, len(nodes) - 1))
    return nodes[j - 1:j + 1]


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

        # water vapor transmittance on the fine grid (full_Twv_lut), built on first use; convolved per sensor
        # description and air-mass nodes, see interp_twv
        self._full_Twv_lut = None
        self._convolved_twv = (None, None, None)

    @property
    def full_Twv_lut(self):
        """Water vapor transmittance on the fine grid at all the air masses (``TWV_AIR_MASSES``) and columns
        (``TWV_TCWV``), built on first use (the precomputed file with the former behavior)"""
        if self._full_Twv_lut is None:
            if behavior.PREVIOUS_BEHAVIOR:
                # Preserve the rework's original precomputed water-vapor table.
                self._full_Twv_lut = xr.open_dataset(WATER_VAPOR_TRANSMITTANCE_FILE)
            else:
                self._full_Twv_lut = self._build_twv_lut()
        return self._full_Twv_lut

    def _build_twv_lut(self, air_masses=TWV_AIR_MASSES):
        r""":math:`T_{wv} = \exp(-M\, W \tau^*_{h2o})` on the fine grid, at ``air_masses`` and ``TWV_TCWV``"""
        air_masses = np.asarray(air_masses, dtype=float)
        air_masses = xr.DataArray(air_masses, coords={"air_mass": air_masses})
        tcwvs = xr.DataArray(TWV_TCWV, coords={"tcwv": TWV_TCWV})
        optical_thickness = self.gas_lut.h2o * tcwvs
        transmittance = np.exp(-air_masses * optical_thickness)
        return transmittance.rename("Twv").to_dataset() # convolve with sensor response in interp_twv()

    def interp_twv(self, wavelengths, sensor_description, air_mass=None):
        """Interpolate Twv to requested sensor bands.

        The sensor convolution can produce the complete acquisition grid while
        a caller asks for only a subset. Interpolation selects that subset on
        its requested ``wl`` coordinate.

        With ``air_mass``, the table holds only the two air-mass nodes around it (:py:func:`air_mass_nodes`): its
        linear interpolation at ``air_mass`` is that of the full table (the convolution is linear), for 1/27 of the
        cost and memory.
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

        nodes = None
        if air_mass is not None and np.isfinite(air_mass):
            nodes = tuple(air_mass_nodes(float(air_mass)))
        # the convolution gives all the sensor bands whatever the requested wavelengths:
        # done once per sensor description and air-mass nodes
        if self._convolved_twv[0] is not sensor_description or self._convolved_twv[1] != nodes:
            table = self.full_Twv_lut if nodes is None else self._build_twv_lut(nodes)
            self._convolved_twv = (sensor_description, nodes, sensor_description.convolve(table.Twv))
        transmittance = self._convolved_twv[2]
        wavelength_dimension = (
            "wl_sensor" if "wl_sensor" in transmittance.dims else "wl"
        )
        transmittance = transmittance.sel(
            {wavelength_dimension: target_wavelengths}
        )
        if wavelength_dimension != "wl" and wavelength_dimension in transmittance.coords:
            transmittance = transmittance.drop_vars(wavelength_dimension)
        return transmittance.rename("Twv").to_dataset()
