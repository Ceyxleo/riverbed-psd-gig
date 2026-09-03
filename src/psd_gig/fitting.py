from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm

from .fit_specs import FunctionSpec
from .progress import ProgressTracker


PARAMETER_COLUMNS = ("fitted_A", "fitted_B", "fitted_C", "fitted_E")
INIT_COLUMNS = ("init_A", "init_B", "init_C", "init_E")


def _sample_xy(
    row: pd.Series,
    d_lookup: pd.DataFrame,
    size_columns: list[str],
    *,
    x_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(row[size_columns], errors="coerce").dropna()
    d = d_lookup.loc[y.index, "d"].to_numpy(dtype=float)
    x = np.log2(d) if x_transform == "log2" else d
    return x, y.to_numpy(dtype=float)


def _metric_row(
    function,
    x: np.ndarray,
    y: np.ndarray,
    parameters: np.ndarray,
    n_params: int,
) -> dict[str, float]:
    fit_y = function(x, *parameters)
    if not np.all(np.isfinite(fit_y)):
        raise ValueError("Fitted values contain non-finite values.")
    mse = mean_squared_error(y, fit_y)
    if not np.isfinite(mse):
        raise ValueError("Mean squared error is non-finite.")
    rmse = np.sqrt(mse)
    n = len(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        aic = n * np.log(mse) + 2 * n_params
        bic = n * np.log(mse) + n_params * np.log(n)
    return {
        "RMSE": rmse,
        "R^2": r2_score(y, fit_y),
        "AIC": aic,
        "BIC": bic,
    }


def _failed_metrics(n_params: int) -> dict[str, float]:
    row = {"RMSE": np.nan, "R^2": np.nan, "AIC": np.nan, "BIC": np.nan}
    for column in PARAMETER_COLUMNS[:n_params]:
        row[column] = np.nan
    return row


def fit_one_sample(
    spec: FunctionSpec,
    x: np.ndarray,
    y: np.ndarray,
    *,
    p0: tuple[float, ...] | None,
    retry_p0: tuple[float, ...] | None = None,
    maxfev: int,
) -> dict[str, float]:
    try:
        if p0 is None:
            parameters, _ = curve_fit(spec.func, x, y, maxfev=maxfev)
        else:
            parameters, _ = curve_fit(spec.func, x, y, p0=p0, maxfev=maxfev)
    except Exception:
        if retry_p0 is None:
            return _failed_metrics(spec.n_params)
        try:
            parameters, _ = curve_fit(spec.func, x, y, p0=retry_p0, maxfev=maxfev)
        except Exception:
            return _failed_metrics(spec.n_params)

    try:
        row = _metric_row(spec.func, x, y, parameters, spec.n_params)
    except Exception:
        return _failed_metrics(spec.n_params)
    for column, value in zip(PARAMETER_COLUMNS, parameters, strict=False):
        row[column] = value
    return row


def base_fit_dataframe(
    spec: FunctionSpec,
    data: pd.DataFrame,
    d_lookup: pd.DataFrame,
    size_columns: list[str],
    *,
    maxfev: int,
    progress: bool,
    tracker: ProgressTracker | None = None,
    progress_log_every: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_samples = len(data)
    iterator = tqdm(data.iterrows(), total=total_samples, disable=not progress, desc=spec.id)
    if tracker is not None:
        tracker.record(
            "stage_started",
            stage="base",
            function_number=spec.number,
            function=spec.label,
            completed_samples=0,
            total_samples=total_samples,
            completed_curve_fits=0,
            total_curve_fits=total_samples,
            message="starting direct curve_fit",
        )
    for completed, (_, sample) in enumerate(iterator, start=1):
        if tracker is not None:
            tracker.record(
                "sample_started",
                stage="base",
                function_number=spec.number,
                function=spec.label,
                completed_samples=completed - 1,
                total_samples=total_samples,
                completed_curve_fits=completed - 1,
                total_curve_fits=total_samples,
                current_site_no=sample["site_no"],
                current_sample_ID=sample["sample_ID"],
                append_event=False,
            )
        x, y = _sample_xy(sample, d_lookup, size_columns, x_transform=spec.base_x_transform)
        fit = fit_one_sample(
            spec,
            x,
            y,
            p0=spec.base_initial_guess,
            retry_p0=spec.base_retry_initial_guess,
            maxfev=maxfev,
        )
        rows.append({"site_no": sample["site_no"], "sample_ID": sample["sample_ID"], **fit})
        if tracker is not None and (completed == total_samples or completed % progress_log_every == 0):
            tracker.record(
                "stage_progress",
                stage="base",
                function_number=spec.number,
                function=spec.label,
                completed_samples=completed,
                total_samples=total_samples,
                completed_curve_fits=completed,
                total_curve_fits=total_samples,
                current_site_no=sample["site_no"],
                current_sample_ID=sample["sample_ID"],
            )
    if tracker is not None:
        tracker.record(
            "stage_finished",
            stage="base",
            function_number=spec.number,
            function=spec.label,
            completed_samples=total_samples,
            total_samples=total_samples,
            completed_curve_fits=total_samples,
            total_curve_fits=total_samples,
            message="finished direct curve_fit",
        )
    return pd.DataFrame(rows)


def _write_csv_chunk(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a" if append else "w", header=not append, index=False)


def _select_best_grid_row(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_df = pd.DataFrame(sample_rows)
    best = sample_df.sort_values("BIC", ascending=True, na_position="last").iloc[0]
    return best.to_dict()


def grid_search_dataframe(
    spec: FunctionSpec,
    data: pd.DataFrame,
    d_lookup: pd.DataFrame,
    size_columns: list[str],
    *,
    maxfev: int,
    progress: bool,
    grid_all_path: Path | None,
    write_grid_search_all: bool,
    grid_chunk_samples: int,
    tracker: ProgressTracker | None = None,
    progress_log_every: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if spec.grid is None:
        raise ValueError(f"{spec.id} does not define a grid search.")

    init_values = spec.grid.initial_values()
    total_samples = len(data)
    total_curve_fits = total_samples * len(init_values)
    best_rows: list[dict[str, Any]] = []
    grid_rows_buffer: list[dict[str, Any]] = []
    wrote_grid_file = False

    if write_grid_search_all and grid_all_path is not None and grid_all_path.exists():
        grid_all_path.unlink()

    iterator = tqdm(data.iterrows(), total=total_samples, disable=not progress, desc=f"{spec.id} grid")
    if tracker is not None:
        tracker.record(
            "stage_started",
            stage="grid",
            function_number=spec.number,
            function=spec.label,
            completed_samples=0,
            total_samples=total_samples,
            completed_curve_fits=0,
            total_curve_fits=total_curve_fits,
            message=f"starting grid search with {len(init_values)} initial guesses per sample",
        )
    for completed, (_, sample) in enumerate(iterator, start=1):
        completed_before_sample = (completed - 1) * len(init_values)
        if tracker is not None:
            tracker.record(
                "sample_started",
                stage="grid",
                function_number=spec.number,
                function=spec.label,
                completed_samples=completed - 1,
                total_samples=total_samples,
                completed_curve_fits=completed_before_sample,
                total_curve_fits=total_curve_fits,
                current_site_no=sample["site_no"],
                current_sample_ID=sample["sample_ID"],
                message=f"fitting {len(init_values)} initial guesses for current sample",
                append_event=False,
            )
        x, y = _sample_xy(sample, d_lookup, size_columns, x_transform="linear")
        sample_rows = []
        for initial_guess in init_values:
            fit = fit_one_sample(spec, x, y, p0=initial_guess, retry_p0=None, maxfev=maxfev)
            row = {
                "site_no": sample["site_no"],
                "sample_ID": sample["sample_ID"],
                **{column: value for column, value in zip(INIT_COLUMNS, initial_guess, strict=False)},
                **fit,
            }
            sample_rows.append(row)
        best_rows.append(_select_best_grid_row(sample_rows))

        if write_grid_search_all and grid_all_path is not None:
            grid_rows_buffer.extend(sample_rows)
            if len(grid_rows_buffer) >= grid_chunk_samples * len(init_values):
                _write_csv_chunk(grid_all_path, grid_rows_buffer, append=wrote_grid_file)
                wrote_grid_file = True
                grid_rows_buffer = []

        if tracker is not None and (completed == total_samples or completed % progress_log_every == 0):
            tracker.record(
                "stage_progress",
                stage="grid",
                function_number=spec.number,
                function=spec.label,
                completed_samples=completed,
                total_samples=total_samples,
                completed_curve_fits=completed * len(init_values),
                total_curve_fits=total_curve_fits,
                current_site_no=sample["site_no"],
                current_sample_ID=sample["sample_ID"],
            )

    if write_grid_search_all and grid_all_path is not None:
        _write_csv_chunk(grid_all_path, grid_rows_buffer, append=wrote_grid_file)

    if tracker is not None:
        tracker.record(
            "stage_finished",
            stage="grid",
            function_number=spec.number,
            function=spec.label,
            completed_samples=total_samples,
            total_samples=total_samples,
            completed_curve_fits=total_curve_fits,
            total_curve_fits=total_curve_fits,
            message="finished grid search",
        )

    final_df = pd.DataFrame(best_rows)
    grid_all_df = None
    if write_grid_search_all and grid_all_path is not None and grid_all_path.exists():
        grid_all_df = pd.read_csv(grid_all_path, dtype={"site_no": str, "sample_ID": str})
    return final_df, grid_all_df
