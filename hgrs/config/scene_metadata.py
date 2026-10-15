"""Normalized acquisition metadata produced by a sensor driver."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SceneMetadata:
    """Platform-neutral acquisition facts for a single scene.

    ``source_georeferencing`` retains source-specific coordinates or
    georeferencing parameters when useful. The raster dataset owns its current
    grid and CRS after reprojection. Platform-specific metadata can be kept in
    ``raw`` without making it part of the normalized processing interface.
    """

    platform: str
    acquisition_time: Any
    solar_zenith: Any
    viewing_zenith: Any
    relative_azimuth: Any
    source_georeferencing: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def from_raster(cls, raster: Any) -> "SceneMetadata":
        """Build complete metadata from a driver-produced raster dataset."""
        description = raster.attrs.get("description")
        if not description:
            raise ValueError("Raster must provide a description identifying its source")

        x_bounds = (float(raster.x.min().values), float(raster.x.max().values))
        y_bounds = (float(raster.y.min().values), float(raster.y.max().values))
        source_georeferencing = {
            "x_bounds": x_bounds,
            "y_bounds": y_bounds,
        }
        try:
            crs = raster.rio.crs
        except (AttributeError, RuntimeError):
            crs = None
        if crs is not None:
            source_georeferencing["crs"] = str(crs)

        return cls(
            platform=str(raster.attrs.get("platform", description.split()[0])),
            acquisition_time=raster.coords["time"],
            solar_zenith=raster["sza"],
            viewing_zenith=raster["vza"],
            relative_azimuth=raster["raa"],
            source_georeferencing=source_georeferencing,
            raw={
                "description": description,
                "product_name": raster.attrs.get("L1C_product_name", ""),
            },
        )
