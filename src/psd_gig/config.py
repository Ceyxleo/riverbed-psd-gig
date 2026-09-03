from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root_from_config(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path.resolve())
    config["_project_root"] = str(project_root_from_config(path))
    return config


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def workflow_dirs(config: dict[str, Any]) -> dict[str, Path]:
    output_root = resolve_project_path(config, config["paths"]["output_root"])
    dirs = {
        "root": output_root,
        "base": output_root / "base_fits",
        "grid_all": output_root / "grid_search_all",
        "final": output_root / "fitted_functions",
        "logs": output_root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
