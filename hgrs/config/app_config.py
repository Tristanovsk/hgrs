"""Shared application configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import yaml

from .atmospheric_correction import AtmosphericCorrectionConfig
from . import behavior


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
            data_root=Path(paths.get("data_root", "data")),
            lut_dir=Path(paths.get("lut_dir", ".")),
            lut_files=lut_files,
            atmospheric_correction=AtmosphericCorrectionConfig.from_mapping(
                config.get("atmospheric_correction", {})
            ),
        )

    def resolve_lut(self, name: str) -> Path:
        """Resolve a configured LUT key beneath the data root and LUT folder."""
        return self.data_root / self.lut_dir / self.lut_files[name]

def get_app_config() -> AppConfig:
    """Load the previous or current packaged defaults selected by behavior."""
    config_name = (
        "old_config.yml" if behavior.PREVIOUS_BEHAVIOR else "default_config.yml"
    )
    config_resource = files("hgrs.config").joinpath(config_name)
    with config_resource.open("r", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream) or {}
    return AppConfig.from_mapping(raw_config)
