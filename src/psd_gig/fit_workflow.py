"""Fit every candidate function to every sample and write one file per function.

Grid search is the default and the only mode used for the published results: for each
sample the fit is repeated from each initial guess on the function's grid and the
lowest-BIC result is kept. Output goes to `<output_root>/fitted_functions/`.

`mode="base"` runs the single-guess fit instead and writes to `<output_root>/base_fits/`,
so the two never overwrite one another.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, workflow_dirs
from .fit_data import load_diameter_lookup, load_input_data, write_data_inventory
from .fit_specs import select_function_specs
from .fitting import base_fit_dataframe, grid_search_dataframe
from .progress import ProgressTracker

#: Column order written for every fit file.
LEAD_COLUMNS = ("site_no", "sample_ID")
METRIC_COLUMNS = ("RMSE", "R2", "AIC", "BIC")


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={"R^2": "R2"})
    order = list(LEAD_COLUMNS)
    order += [c for c in ("init_A", "init_B", "init_C") if c in frame.columns]
    order += [c for c in ("fitted_A", "fitted_B", "fitted_C") if c in frame.columns]
    order += [c for c in METRIC_COLUMNS if c in frame.columns]
    return frame[order]


def _fit_one(
    spec, data, d_lookup, size_columns, *, mode: str, maxfev: int,
    progress: bool, tracker=None, progress_log_every: int = 100,
    grid_all_path: Path | None = None, grid_chunk_samples: int = 100,
) -> pd.DataFrame:
    """Fit one function to every sample and return the canonical frame."""
    if mode == "grid":
        frame, _ = grid_search_dataframe(
            spec, data, d_lookup, size_columns,
            maxfev=maxfev, progress=progress,
            grid_all_path=grid_all_path,
            write_grid_search_all=grid_all_path is not None,
            grid_chunk_samples=grid_chunk_samples,
            tracker=tracker, progress_log_every=progress_log_every,
        )
    else:
        frame = base_fit_dataframe(
            spec, data, d_lookup, size_columns,
            maxfev=maxfev, progress=progress,
            tracker=tracker, progress_log_every=progress_log_every,
        )
    return _canonical_columns(frame)


def _fit_one_to_file(payload: tuple) -> tuple[int, str, int]:
    """Worker entry point: fit one function and write it. Returns (number, path, rows)."""
    spec, data, d_lookup, size_columns, path, mode, maxfev = payload
    frame = _fit_one(spec, data, d_lookup, size_columns, mode=mode, maxfev=maxfev,
                     progress=False, tracker=None)
    frame.to_csv(path, index=False)
    return spec.number, str(path), len(frame)


def run_fit_workflow(
    config_path: str | Path,
    *,
    function_selectors: list[str] | None = None,
    mode: str = "grid",
    overwrite: bool | None = None,
    limit_samples: int | None = None,
    jobs: int = 1,
) -> dict[str, Path]:
    if mode not in {"grid", "base"}:
        raise ValueError(f"mode must be 'grid' or 'base', not {mode!r}")

    config = load_config(config_path)
    runtime = config.setdefault("runtime", {})
    if overwrite is not None:
        runtime["overwrite"] = overwrite
    if limit_samples is not None:
        runtime["limit_samples"] = limit_samples

    dirs = workflow_dirs(config, mode)
    data, size_columns = load_input_data(config)
    d_lookup = load_diameter_lookup(config, size_columns)
    specs = select_function_specs(function_selectors)

    write_data_inventory(config, dirs["logs"] / "data_inventory.csv", data, size_columns)

    maxfev = int(runtime.get("maxfev", 20000))
    progress = bool(runtime.get("progress", True))
    overwrite_files = bool(runtime.get("overwrite", False))
    # The per-guess dump is a diagnostic, off by default: it is ~50x the size of the result.
    write_grid_search_all = bool(runtime.get("write_grid_search_all", False))
    grid_chunk_samples = int(runtime.get("grid_chunk_samples", 100))
    progress_log_every = int(runtime.get("progress_log_every", 100))
    tracker = ProgressTracker(dirs["logs"], enabled=bool(runtime.get("progress_tracker", True)))
    tracker.record(
        "run_started",
        message=f"mode={mode}; functions={','.join(spec.label for spec in specs)}",
    )

    todo = [spec for spec in specs
            if overwrite_files or not (dirs["fits"] / spec.output_name).exists()]
    skipped = [spec for spec in specs if spec not in todo]

    if jobs > 1 and len(todo) > 1:
        if write_grid_search_all:
            print("note: write_grid_search_all needs --jobs 1; the per-guess dump is skipped.")
        # One worker per function. Workers keep no tracker or progress bar of their own,
        # so the shared log files are written by this process alone.
        payloads = [(spec, data, d_lookup, size_columns, dirs["fits"] / spec.output_name,
                     mode, maxfev) for spec in todo]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for number, path, rows in pool.map(_fit_one_to_file, payloads):
                tracker.record("function_finished",
                               message=f"F{number:02d} -> {Path(path).name} ({rows} rows)")
                print(f"  F{number:02d} {Path(path).name}  {rows:,} rows", flush=True)
    else:
        for spec in todo:
            grid_all_path = dirs["grid_all"] / f"grid_all_{spec.id}.csv"
            if write_grid_search_all:
                grid_all_path.parent.mkdir(parents=True, exist_ok=True)
            frame = _fit_one(
                spec, data, d_lookup, size_columns, mode=mode, maxfev=maxfev,
                progress=progress, tracker=tracker,
                progress_log_every=progress_log_every,
                grid_all_path=grid_all_path if write_grid_search_all else None,
                grid_chunk_samples=grid_chunk_samples,
            )
            frame.to_csv(dirs["fits"] / spec.output_name, index=False)

    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        path = dirs["fits"] / spec.output_name
        n_guesses = len(spec.grid) if mode == "grid" else 1
        status = "skipped_existing" if spec in skipped else "written"
        manifest_rows.append({
            "function_number": spec.number,
            "function": spec.label,
            "n_params": spec.n_params,
            "mode": mode,
            "n_initial_guesses": n_guesses,
            "maxfev": maxfev,
            "status": status,
            "file": str(path),
        })

    manifest_path = dirs["logs"] / f"fit_manifest_{mode}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    tracker.record("run_finished", message=f"manifest={manifest_path}")
    return {
        "output_root": dirs["root"],
        "fits": dirs["fits"],
        "manifest": manifest_path,
        "data_inventory": dirs["logs"] / "data_inventory.csv",
    }
