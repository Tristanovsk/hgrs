"""Settings for the atmospheric-correction product and its solvers."""

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import numpy as np
import xarray as xr


def _dataclass_values(config, cls, tuple_fields=(), range_fields=()):
    known = {item.name for item in fields(cls)}
    unknown = set(config) - known
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} setting(s): "
            + ", ".join(sorted(unknown))
        )

    values = dict(config)
    for name in tuple_fields:
        if name in values:
            values[name] = tuple(values[name])
    for name in range_fields:
        if name in values:
            values[name] = tuple(tuple(bounds) for bounds in values[name])
    return values


@dataclass(frozen=True)
class ProductCorrectionConfig:
    """Settings consumed directly by :class:`hgrs.Product`."""

    sunglint_wavelengths_nm: tuple[float, float] = (2150.0, 2250.0)
    green_mask_wavelengths_nm: tuple[float, float] = (540.0, 570.0)
    nir_mask_wavelengths_nm: tuple[float, float] = (850.0, 882.0)
    green_swir_mask_wavelengths_nm: tuple[float, float] = (1580.0, 1650.0)
    x_coarsen: int = 20
    y_coarsen: int = 20
    block_size: int = 2
    water_pixel_percentage: int = 20
    angle_rounding_digits: int = 1
    sunglint_threshold: float = 0.11
    ndwi_threshold: float = 0.01
    green_swir_index_threshold: float = 0.1
    gas_absorption_scattering_coefficient: float = 0.35
    prisma_land_mask_enabled: bool = True
    prisma_land_valid_classes: tuple[int, ...] = (0, 5, 6)
    prisma_land_forbidden_classes: tuple[int, ...] = (1, 2, 3, 4, 10)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ProductCorrectionConfig":
        tuple_fields = (
            "sunglint_wavelengths_nm",
            "green_mask_wavelengths_nm",
            "nir_mask_wavelengths_nm",
            "green_swir_mask_wavelengths_nm",
            "prisma_land_valid_classes",
            "prisma_land_forbidden_classes",
        )
        return cls(**_dataclass_values(config, cls, tuple_fields=tuple_fields))


@dataclass(frozen=True)
class WaterVaporConfig:
    """Water-vapor retrieval settings for one product."""

    wavelengths_nm: tuple[float, float] = (800.0, 1300.0)
    first_guess: tuple[float, float, float] = (2.0, -0.04, 0.1)
    smoothing_method: str = "none"
    smoothing_window: tuple[int, int] = (5, 5)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "WaterVaporConfig":
        return cls(
            **_dataclass_values(
                config,
                cls,
                tuple_fields=("wavelengths_nm", "first_guess", "smoothing_window"),
            )
        )


@dataclass(frozen=True)
class WaterParameters:
    """Water masking and retrieval settings grouped for a single product."""

    mask: ProductCorrectionConfig
    vapor: WaterVaporConfig

    @property
    def wl_sunglint(self) -> slice:
        return slice(*self.mask.sunglint_wavelengths_nm)

    @property
    def wl_green(self) -> slice:
        return slice(*self.mask.green_mask_wavelengths_nm)

    @property
    def wl_nir(self) -> slice:
        return slice(*self.mask.nir_mask_wavelengths_nm)

    @property
    def wl_1600(self) -> slice:
        return slice(*self.mask.green_swir_mask_wavelengths_nm)

    @property
    def wl_water_vapor(self) -> slice:
        return slice(*self.vapor.wavelengths_nm)


@dataclass(frozen=True)
class AerosolCorrectionConfig:
    """Aerosol solver settings plus CAMS-derived values for one scene."""

    fit_wavelengths_nm: tuple[float, ...] = (
        1000.0, 1050.0, 1075.0, 1100.0, 1200.0, 1300.0,
        1600.0, 1650.0, 1700.0, 2150.0, 2200.0, 2250.0,
    )
    nonnegative_reflectance_wavelengths_nm: tuple[float, ...] = (
        430.0, 490.0, 560.0, 650.0, 750.0, 800.0, 865.0, 1020.0,
    )
    excluded_wavelength_ranges_nm: tuple[tuple[float, float], ...] = (
        (935.0, 967.0), (1105.0, 1170.0), (1320.0, 1490.0),
        (1778.0, 2033.0), (2465.0, 2550.0),
    )
    smoothing_method: str = "weighted_local"
    smoothing_window: tuple[int, int] = (1, 1)
    smoothing_footprint: tuple[int, int] = (3, 3)
    solver_max_iterations: int = 10
    solver_ftol: float | None = None
    solver_warm_start: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AerosolCorrectionConfig":
        return cls(**_dataclass_values(
            config,
            cls,
            tuple_fields=(
                "fit_wavelengths_nm",
                "nonnegative_reflectance_wavelengths_nm",
                "smoothing_window",
                "smoothing_footprint",
            ),
            range_fields=("excluded_wavelength_ranges_nm",),
        ))


@dataclass(frozen=True)
class AerosolParameters:
    """Complete, self-contained aerosol solver inputs for one scene."""

    fit_wavelengths_nm: tuple[float, ...]
    nonnegative_reflectance_wavelengths_nm: tuple[float, ...]
    excluded_wavelength_ranges_nm: tuple[tuple[float, float], ...]
    aerosol_model: str
    aod550_max: float
    aod550_initial: float
    solver_max_iterations: int = 10
    solver_ftol: float | None = None
    solver_warm_start: bool = True
    aod550_min: float = 0.002
    smoothing_method: str = "weighted_local"
    smoothing_window: tuple[int, int] = (1, 1)
    smoothing_footprint: tuple[int, int] = (3, 3)

    @property
    def aod550_bounds(self) -> tuple[float, float]:
        return self.aod550_min, self.aod550_max

    @classmethod
    def from_cams(
        cls,
        cams_data: xr.Dataset,
        aerosol_lut: xr.Dataset,
        config: AerosolCorrectionConfig,
    ) -> "AerosolParameters":
        """Resolve static settings and CAMS priors into a complete scene state."""
        cams_wavelengths = [469, 550, 670, 865, 1240]
        cams_aod = cams_data[
            [f"aod{wavelength}" for wavelength in cams_wavelengths]
        ].to_array(dim="wl")
        cams_aod = cams_aod.assign_coords(
            wl=cams_aod.wl.str.replace("aod", "").astype(float)
        )

        lut_aod = aerosol_lut.aot.sel(aot_ref=1).interp(wl=cams_aod.wl)
        model_index = np.abs(
            (cams_aod / cams_data.aod550) - lut_aod
        ).sum("wl").argmin()
        cams_model = str(aerosol_lut.model.values[model_index])

        prior_mean = float(cams_data.aod550.mean().values)
        prior_std = float(cams_data.aod550.std().values)
        prior_std = np.max([prior_std, 0.2 * prior_mean + 0.05])
        aod550_max = prior_mean + 2 * prior_std

        return cls(
            fit_wavelengths_nm=config.fit_wavelengths_nm,
            nonnegative_reflectance_wavelengths_nm=(
                config.nonnegative_reflectance_wavelengths_nm
            ),
            excluded_wavelength_ranges_nm=config.excluded_wavelength_ranges_nm,
            aerosol_model=cams_model,
            aod550_max=aod550_max,
            aod550_initial=prior_mean,
            solver_max_iterations=config.solver_max_iterations,
            solver_ftol=config.solver_ftol,
            solver_warm_start=config.solver_warm_start,
            smoothing_method=config.smoothing_method,
            smoothing_window=config.smoothing_window,
            smoothing_footprint=config.smoothing_footprint,
        )


@dataclass(frozen=True)
class AtmosphericCorrectionConfig:
    """Grouped configuration for Product, WaterVapor and Aerosol processing."""

    product: ProductCorrectionConfig = field(default_factory=ProductCorrectionConfig)
    water_vapor: WaterVaporConfig = field(default_factory=WaterVaporConfig)
    aerosol: AerosolCorrectionConfig = field(default_factory=AerosolCorrectionConfig)
    bidirectional_transmittance_global: bool = True

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any]
    ) -> "AtmosphericCorrectionConfig":
        groups = {"product", "water_vapor", "aerosol"}
        if set(config).issubset(groups | {"bidirectional_transmittance_global"}):
            grouped = dict(config)
        else:
            grouped = cls._group_legacy_mapping(config)

        bidirectional_global = grouped.pop("bidirectional_transmittance_global", True)
        unknown = set(grouped) - groups
        if unknown:
            raise ValueError(
                "Unknown atmospheric_correction section(s): "
                + ", ".join(sorted(unknown))
            )
        return cls(
            product=ProductCorrectionConfig.from_mapping(grouped.get("product", {})),
            water_vapor=WaterVaporConfig.from_mapping(
                grouped.get("water_vapor", {})
            ),
            aerosol=AerosolCorrectionConfig.from_mapping(grouped.get("aerosol", {})),
            bidirectional_transmittance_global=bidirectional_global,
        )

    @staticmethod
    def _group_legacy_mapping(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Accept the former flat YAML layout while mapping values to subobjects."""
        mapping = {
            "water_vapor_wavelengths_nm": ("water_vapor", "wavelengths_nm"),
            "aerosol_fit_wavelengths_nm": ("aerosol", "fit_wavelengths_nm"),
            "nonnegative_reflectance_wavelengths_nm": (
                "aerosol", "nonnegative_reflectance_wavelengths_nm"
            ),
            "excluded_wavelength_ranges_nm": (
                "aerosol", "excluded_wavelength_ranges_nm"
            ),
            "sunglint_wavelengths_nm": ("product", "sunglint_wavelengths_nm"),
            "green_mask_wavelengths_nm": ("product", "green_mask_wavelengths_nm"),
            "nir_mask_wavelengths_nm": ("product", "nir_mask_wavelengths_nm"),
            "green_swir_mask_wavelengths_nm": (
                "product", "green_swir_mask_wavelengths_nm"
            ),
            "x_coarsen": ("product", "x_coarsen"),
            "y_coarsen": ("product", "y_coarsen"),
            "block_size": ("product", "block_size"),
            "water_pixel_percentage": ("product", "water_pixel_percentage"),
            "angle_rounding_digits": ("product", "angle_rounding_digits"),
            "sunglint_threshold": ("product", "sunglint_threshold"),
            "ndwi_threshold": ("product", "ndwi_threshold"),
            "green_swir_index_threshold": ("product", "green_swir_index_threshold"),
            "prisma_land_mask_enabled": ("product", "prisma_land_mask_enabled"),
            "prisma_land_valid_classes": ("product", "prisma_land_valid_classes"),
            "prisma_land_forbidden_classes": ("product", "prisma_land_forbidden_classes"),
            "aerosol_absorption_scattering_coefficient": (
                "product", "gas_absorption_scattering_coefficient"
            ),
            "bidirectional_transmittance_global": (
                "top_level", "bidirectional_transmittance_global"
            ),
        }
        grouped = {"product": {}, "water_vapor": {}, "aerosol": {}}
        top_level = {}
        unknown = set(config) - set(mapping)
        if unknown:
            raise ValueError(
                "Unknown atmospheric_correction setting(s): "
                + ", ".join(sorted(unknown))
            )
        for name, value in config.items():
            section, field_name = mapping[name]
            if section == "top_level":
                top_level[field_name] = value
            else:
                grouped[section][field_name] = value
        grouped.update(top_level)
        return grouped
