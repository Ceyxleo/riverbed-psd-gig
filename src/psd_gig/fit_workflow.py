from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, workflow_dirs
from .fit_data import load_diameter_lookup, load_input_data, write_data_inventory
from .fit_specs import FunctionSpec, select_function_specs
from .fitting import base_fit_dataframe, grid_search_dataframe
from .progress import ProgressTracker


def _write_or_read(path: Path, dataframe: pd.DataFrame, *, overwrite: bool) -> tuple[pd.DataFrame, str]:
    if path.exists() and not overwrite:
        return pd.read_csv(path, dtype={"site_no": str, "sample_ID": str}), "skipped_existing"
    dataframe.to_csv(path, index=False)
    return dataframe, "written"


def _copy_dataframe_to_final(path: Path, dataframe: pd.DataFrame, *, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return "skipped_existing"
    dataframe.to_csv(path, index=False)
    return "written"


def _append_filename_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def run_fit_workflow(
    config_path: str | Path,
    *,
    function_selectors: list[str] | None = None,
    stages: list[str] | None = None,
    overwrite: bool | None = None,
    limit_samples: int | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    runtime = config.setdefault("runtime", {})
    if overwrite is not None:
        runtime["overwrite"] = overwrite
    if limit_samples is not None:
        runtime["limit_samples"] = limit_samples

    selected_stages = set(stages or ["base", "grid", "final"])
    dirs = workflow_dirs(config)
    data, size_columns = load_input_data(config)
    d_lookup = load_diameter_lookup(config, size_columns)
    specs = select_function_specs(function_selectors)

    write_data_inventory(config, dirs["logs"] / "data_inventory.csv", data, size_columns)

    maxfev = int(runtime.get("maxfev", 1000000))
    progress = bool(runtime.get("progress", True))
    overwrite_files = bool(runtime.get("overwrite", False))
    write_grid_search_all = bool(runtime.get("write_grid_search_all", True))
    grid_chunk_samples = int(runtime.get("grid_chunk_samples", 100))
    progress_log_every = int(runtime.get("progress_log_every", 100))
    output_filename_suffix = str(runtime.get("output_filename_suffix") or "")
    tracker = ProgressTracker(
        dirs["logs"],
        enabled=bool(runtime.get("progress_tracker", True)),
    )
    tracker.record(
        "run_started",
        message=(
            f"stages={','.join(sorted(selected_stages))}; "
            f"functions={','.join(spec.label for spec in specs)}"
        ),
    )

    manifest_rows: list[dict[str, Any]] = []
    outputs: dict[str, Path] = {"output_root": dirs["root"]}

    for spec in specs:
        base_path = _append_filename_suffix(dirs["base"] / spec.base_output_name, output_filename_suffix)
        final_path = _append_filename_suffix(dirs["final"] / spec.final_output_name, output_filename_suffix)
        grid_path = (
            _append_filename_suffix(dirs["grid_all"] / spec.grid.grid_output_name, output_filename_suffix)
            if spec.grid
            else None
        )

        base_df: pd.DataFrame | None = None
        base_status = "not_requested"
        if "base" in selected_stages:
            if base_path.exists() and not overwrite_files:
                base_df = pd.read_csv(base_path, dtype={"site_no": str, "sample_ID": str})
                base_status = "skipped_existing"
            else:
                base_df = base_fit_dataframe(
                    spec,
                    data,
                    d_lookup,
                    size_columns,
                    maxfev=maxfev,
                    progress=progress,
                    tracker=tracker,
                    progress_log_every=progress_log_every,
                )
                base_df, base_status = _write_or_read(base_path, base_df, overwrite=overwrite_files)

        grid_final_df: pd.DataFrame | None = None
        grid_status = "not_applicable" if spec.grid is None else "not_requested"
        if spec.grid is not None and "grid" in selected_stages:
            if final_path.exists() and grid_path and grid_path.exists() and not overwrite_files:
                grid_final_df = pd.read_csv(final_path, dtype={"site_no": str, "sample_ID": str})
                grid_status = "skipped_existing"
            else:
                grid_final_df, _ = grid_search_dataframe(
                    spec,
                    data,
                    d_lookup,
                    size_columns,
                    maxfev=maxfev,
                    progress=progress,
                    grid_all_path=grid_path,
                    write_grid_search_all=write_grid_search_all,
                    grid_chunk_samples=grid_chunk_samples,
                    tracker=tracker,
                    progress_log_every=progress_log_every,
                )
                grid_final_df.to_csv(final_path, index=False)
                grid_status = "written"

        final_status = "not_requested"
        if "final" in selected_stages:
            if spec.grid is not None:
                if grid_final_df is not None:
                    if grid_status == "written":
                        final_status = "written_from_grid"
                    else:
                        final_status = _copy_dataframe_to_final(final_path, grid_final_df, overwrite=overwrite_files)
                elif final_path.exists():
                    final_status = "kept_existing_grid_final"
                else:
                    final_status = "missing_grid_final"
            else:
                if base_df is None and base_path.exists():
                    base_df = pd.read_csv(base_path, dtype={"site_no": str, "sample_ID": str})
                if base_df is not None:
                    final_status = _copy_dataframe_to_final(final_path, base_df, overwrite=overwrite_files)
                else:
                    final_status = "missing_base_fit"

        manifest_rows.append(
            {
                "function_number": spec.number,
                "function": spec.label,
                "n_params": spec.n_params,
                "has_grid_search": spec.grid is not None,
                "base_status": base_status,
                "grid_status": grid_status,
                "final_status": final_status,
                "base_file": str(base_path),
                "grid_search_all_file": str(grid_path) if grid_path else "",
                "final_file": str(final_path),
            }
        )

    manifest_path = dirs["logs"] / "fit_workflow_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    tracker.record(
        "run_finished",
        message=f"manifest={manifest_path}",
    )
    outputs["manifest"] = manifest_path
    outputs["data_inventory"] = dirs["logs"] / "data_inventory.csv"
    outputs["progress_latest"] = dirs["logs"] / "progress_latest.csv"
    outputs["progress_events"] = dirs["logs"] / "progress_events.jsonl"
    outputs["progress_by_function"] = dirs["logs"] / "progress_by_function.csv"
    return outputs
