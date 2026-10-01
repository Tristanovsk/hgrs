"""Sensor spectral operators used by an hGRS processing run."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SensorDescription:
    """Sensor identity, acquisition convolution, and lower-resolution interpolation."""

    name: str
    sensor_mod: Any
    sensor_mod_lr: Any

    @classmethod
    def default(cls, raster, *, name):
        """Build the standard sensor operators from a raster's spectral metadata."""
        from ..spectral_sensitivity import BaselineInterp, Gaussian

        wavelengths = raster.wl.values
        return cls(
            name=name,
            sensor_mod=Gaussian(wavelengths, raster.fwhm.values),
            sensor_mod_lr=BaselineInterp(
                wavelengths,
                dim_wl_sensor="wl",
                inter_mod="quadratic",
            ),
        )

    def convolve(self, signal, **options):
        """Simulate sensor acquisition by convolving a high-resolution spectrum."""
        return self.sensor_mod.convolve(signal, **options)

    def interpolate(self, signal, **options):
        """Interpolate lower-resolution data onto the sensor wavelength grid."""
        return self.sensor_mod_lr.convolve(signal, **options)
