"""Shared application configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .atmospheric_correction import AtmosphericCorrectionConfig
from .loader import load_config


# Files of the same LUT under another name: the TOA LUT published on Zenodo (shared with GRS) and the
# uncompressed copy used by hGRS up to 2.1 (same I, Ed within 2.3e-5), so that an existing LUT folder
# keeps working whichever of the two names is configured
LUT_ALIASES = {
    "toa_lut_opac_wind_v3.nc": ("toa_lut_opac_wind_up_v3.nc",),
    "toa_lut_opac_wind_up_v3.nc": ("toa_lut_opac_wind_v3.nc",),
}


@dataclass(frozen=True)
class AppConfig:
    """Settings and resource locations shared by processing runs."""

    data_root: Path
    lut_dir: Path = Path("lut")
    lut_files: Mapping[str, str] = field(default_factory=dict)
    atmospheric_correction: AtmosphericCorrectionConfig = field(
        default_factory=AtmosphericCorrectionConfig
    )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AppConfig":
        """Build from the current YAML structure, preserving legacy LUT paths."""
        paths = config.get("path", {})
        lut_files = {
            key: paths[value]
            for key, value in (
                ("toa", "toa_lut"),
                ("transmittance", "trans_lut"),
                ("abs_gas", "abs_gas_file"),
            )
            if value in paths
        }
        return cls(
            data_root=Path(paths["data_root"]),
            lut_dir=Path(paths.get("lut_dir", ".")),
            lut_files=lut_files,
            atmospheric_correction=AtmosphericCorrectionConfig.from_mapping(
                config.get("atmospheric_correction", {})
            ),
        )

    def resolve_lut(self, name: str) -> Path:
        """
        Resolve a configured LUT key beneath the data root and LUT folder.

        When the file is not found under the data root, the same LUT under another name
        (:py:data:`LUT_ALIASES`) is used if it is there, then the LUTs shipped with the package
        (``hgrs/data/lut``, e.g. the gaseous absorption LUT).
        """
        path = self.data_root / self.lut_dir / self.lut_files[name]
        if not path.exists():
            for alias in LUT_ALIASES.get(path.name, ()):
                if path.with_name(alias).is_file():
                    return path.with_name(alias)
            packaged = Path(str(files("hgrs").joinpath("data/lut", self.lut_files[name])))
            if packaged.exists():
                return packaged
        return path

    def check_luts(self):
        """
        Check that the LUT files exist.

        :raises FileNotFoundError: listing the missing LUTs and how to set their folder
        """
        missing = [str(path) for path in map(self.resolve_lut, self.lut_files) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "LUT file(s) not found: " + ", ".join(missing)
                + f". Download them with `hgrs_config --download` (into {self.data_root}), copy "
                "them there, or set their folder with "
                "`hgrs_config --init /path/to/lut` (or path: data_root in the user configuration "
                "file, or HGRS_DATA_ROOT); check the configuration with `hgrs_config`."
            )


def get_app_config() -> AppConfig:
    """Load the configuration: packaged defaults (selected by behavior), user file, environment."""
    return AppConfig.from_mapping(load_config())
