"""Settings for the atmospheric-correction product and its solvers."""

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import numpy as np
import xarray as xr

AEROSOL_PRIORS = ("none", "cams")
WATER_VAPOR_PRIORS = ("none", "cams")
WATER_VAPOR_SURFACES = ("linear", "non_negative")
WATER_VAPOR_OFFSETS = ("none", "saturated_bands")
WATER_VAPOR_PATH_SLOPES = ("fixed", "free")
NONNEGATIVITY_MODES = ("hard", "soft")


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
    # coupling of the gaseous absorption with the scattering (hgrs.gas_coupling): 'path_length' or
    # 'decoupled' (direct-beam transmittance for the whole signal); O2 absorption in the O2-B and O2-A
    # bands: 'lbl_o2b' (line by line in the O2-B band, REPTRAN in the O2-A band), 'lbl' (line by line in
    # both) or 'reptran'. The defaults here are the former behavior; the packaged default_config.yml
    # selects 'path_length' and 'lbl_o2b'.
    gas_coupling: str = "decoupled"
    o2_model: str = "reptran"
    # table of the coefficients of the photon path-length model (None: the table of the package)
    gas_coupling_table: str | None = None
    # EnMAP: correction of the radiometric mismatch of the VNIR and SWIR detectors, SWIR - VNIR estimated per
    # scene over water at 990 nm (detector overlap): 'none', 'swir_ramp' (offset subtracted from the SWIR,
    # tapered linearly to 0 at 2200 nm) or 'vnir_ramp' (offset added to the VNIR, ramped linearly from 0 at
    # 700 nm), see hgrs.drivers.enmap.detector_offset
    enmap_detector_offset: str = "none"
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
    # prior of the water vapor column (Tikhonov term ((W - W_CAMS) / sigma_W)^2 in the cost): 'none' or
    # 'cams', with sigma_W = max(prior_sigma_relative * W_CAMS, prior_sigma_min) (kg m-2)
    prior: str = "none"
    prior_sigma_relative: float = 0.1
    prior_sigma_min: float = 2.0
    # uncertainty of the fitted reflectance, weight of the data term against the prior
    reflectance_sigma: float = 1e-3
    # surface term of the coupled model (gas_coupling: path_length): 'linear' (a wl + b, any sign, 2.0.0) or
    # 'non_negative' (linear between its values at the ends of the fitting range, both >= 0)
    surface: str = "linear"
    # bounds of the scale factor k of the LUT path reflectance in the coupled model ((0, 5) in 2.0.0)
    path_scale_limits: tuple[float, float] = (0.0, 5.0)
    # spectral slope of the path reflectance with surface: non_negative: 'fixed' (shape of the LUT) or 'free'
    # (k R_path wl^-dn, wl in micron, dn in path_slope_limits)
    path_slope: str = "fixed"
    path_slope_limits: tuple[float, float] = (-4.0, 4.0)
    # signal above the water vapour (thin cirrus, stray light), subtracted from the data before the fit:
    # 'none' or 'saturated_bands' (median reflectance of the bands of offset_wavelengths_nm in each mega-pixel)
    offset: str = "none"
    offset_wavelengths_nm: tuple[float, float] = (1360.0, 1395.0)

    def __post_init__(self):
        if self.prior not in WATER_VAPOR_PRIORS:
            raise ValueError(f"water_vapor: prior must be one of {WATER_VAPOR_PRIORS}, not {self.prior!r}")
        if self.surface not in WATER_VAPOR_SURFACES:
            raise ValueError(f"water_vapor: surface must be one of {WATER_VAPOR_SURFACES}, not {self.surface!r}")
        if self.offset not in WATER_VAPOR_OFFSETS:
            raise ValueError(f"water_vapor: offset must be one of {WATER_VAPOR_OFFSETS}, not {self.offset!r}")
        if self.path_slope not in WATER_VAPOR_PATH_SLOPES:
            raise ValueError(f"water_vapor: path_slope must be one of {WATER_VAPOR_PATH_SLOPES}, not {self.path_slope!r}")
        if self.path_slope == "free" and self.surface != "non_negative":
            raise ValueError("water_vapor: path_slope: free requires surface: non_negative")
        if not self.path_slope_limits[0] < self.path_slope_limits[1]:
            raise ValueError("water_vapor: path_slope_limits must be (dn_min, dn_max) with dn_min < dn_max")
        k_min, k_max = self.path_scale_limits
        if not 0 <= k_min < k_max:
            raise ValueError("water_vapor: path_scale_limits must be (k_min, k_max) with 0 <= k_min < k_max")
        if self.prior_sigma_relative < 0 or self.prior_sigma_min <= 0 or self.reflectance_sigma <= 0:
            raise ValueError("water_vapor: prior_sigma_relative must be >= 0, prior_sigma_min and "
                             "reflectance_sigma > 0")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "WaterVaporConfig":
        return cls(
            **_dataclass_values(
                config,
                cls,
                tuple_fields=("wavelengths_nm", "first_guess", "smoothing_window", "path_scale_limits",
                              "offset_wavelengths_nm", "path_slope_limits"),
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
    # prior of the AOT (Tikhonov term ((tau - tau_CAMS) / sigma_CAMS)^2 in the cost): 'none' or 'cams'
    prior: str = "none"
    # uncertainty of the fitted TOA reflectance, weight of the data term against the prior and the penalty
    reflectance_sigma: float = 1e-3
    # non-negativity of the water reflectance: 'hard' (SLSQP constraint) or 'soft' (one-sided quadratic
    # penalty sum(min(0, R - R_sim)^2) / nonnegativity_sigma^2 in the cost)
    nonnegativity: str = "hard"
    nonnegativity_sigma: float = 1e-3

    def __post_init__(self):
        if self.prior not in AEROSOL_PRIORS:
            raise ValueError(f"aerosol: prior must be one of {AEROSOL_PRIORS}, not {self.prior!r}")
        if self.nonnegativity not in NONNEGATIVITY_MODES:
            raise ValueError(
                f"aerosol: nonnegativity must be one of {NONNEGATIVITY_MODES}, not {self.nonnegativity!r}")
        if self.reflectance_sigma <= 0 or self.nonnegativity_sigma <= 0:
            raise ValueError("aerosol: reflectance_sigma and nonnegativity_sigma must be positive")

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
    aod550_prior_std: float = float("nan")
    prior: str = "none"
    reflectance_sigma: float = 1e-3
    nonnegativity: str = "hard"
    nonnegativity_sigma: float = 1e-3

    @property
    def regularized(self) -> bool:
        """True when the cost has a prior or a soft non-negativity term (else the 1.x/2.0 fit)."""
        return self.prior != "none" or self.nonnegativity != "hard"

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
        model_index = int(np.argmin(np.abs(
            (cams_aod / cams_data.aod550) - lut_aod
        ).sum("wl").values))
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
            aod550_prior_std=float(prior_std),
            prior=config.prior,
            reflectance_sigma=config.reflectance_sigma,
            nonnegativity=config.nonnegativity,
            nonnegativity_sigma=config.nonnegativity_sigma,
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
            "gas_coupling": ("product", "gas_coupling"),
            "o2_model": ("product", "o2_model"),
            "gas_coupling_table": ("product", "gas_coupling_table"),
            "enmap_detector_offset": ("product", "enmap_detector_offset"),
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
