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


def workflow_dirs(config: dict[str, Any], mode: str = "grid") -> dict[str, Path]:
    """Output directories for one fitting run.

    Grid search -- the default -- writes to `fitted_functions/`. `--mode base` writes
    the single-guess fits to a separate `base_fits/` folder, so the two never mix.
    """
    output_root = resolve_project_path(config, config["paths"]["output_root"])
    dirs = {
        "root": output_root,
        "fits": output_root / ("fitted_functions" if mode == "grid" else "base_fits"),
        "grid_all": output_root / "grid_search_all",
        "logs": output_root / "logs",
    }
    for key, path in dirs.items():
        if key != "grid_all":
            path.mkdir(parents=True, exist_ok=True)
    return dirs
