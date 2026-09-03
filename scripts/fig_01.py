from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from _figure1_common import (
    AXIS_COLOR,
    DATA_PATH,
    FUNCTION_COLORS,
    FUNCTION_LABELS,
    FUNCTION_ORDER,
    FUNCTION_STYLES,
    OUT_PREFIX,
    SIZE_PATH,
    eval_density,
    load_data,
    load_fits,
    load_size_lookup,
    setup_style,
    style_axis,
)


SAMPLE_OUT_PREFIX = OUT_PREFIX.with_name("Figure_1")
TARGETS = [
    ("P10", "coarse-skewed", 0.10),
    ("zero", "symmetric", None),
    ("P90", "fine-skewed", 0.90),
]
OBS_FACE = "#D9D9D9"
OBS_EDGE = "#7A7A7A"
FULL_XLIM = (1 / 64, 64)
MID_XLIM = (1 / 16, 4)
FULL_XTICKS = np.array([1 / 64, 1 / 16, 1 / 4, 1, 4, 16, 64], dtype=float)
MID_XTICKS = np.array([1 / 16, 1 / 4, 1, 4], dtype=float)
SAMPLE_STYLES = {
    "P10": "-.",
    "zero": "--",
    "P90": "-",
}
BOTTOM_FUNCTION_ORDER = ["Lognormal", "Weibull", "GIG_2p"]


def key_series(df: pd.DataFrame) -> pd.Series:
    return df["site_no"].astype("string") + "|" + df["sample_ID"].astype("Int64").astype("string")


def samples_with_all_fits(data: pd.DataFrame, fits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    available = None
    for fit in fits.values():
        keys = set(key_series(fit.dropna(subset=["sample_ID"])))
        available = keys if available is None else available & keys
    filtered = data.copy()
    filtered["_fit_key"] = key_series(filtered)
    return filtered[filtered["_fit_key"].isin(available)].drop(columns="_fit_key")


def select_representatives(data: pd.DataFrame) -> tuple[list[pd.Series], dict[str, float]]:
    skew = data["skew_FW"]
    median_d50 = data.loc[data["D50"] > 0, "D50"].median()
    thresholds = {
        "P10": float(skew.quantile(0.10)),
        "zero": 0.0,
        "P90": float(skew.quantile(0.90)),
    }
    selected = []
    for target_name, label, quantile in TARGETS:
        target_value = thresholds[target_name]
        candidates = data.assign(_distance=(data["skew_FW"] - target_value).abs())
        candidates["_d50_distance"] = np.abs(np.log(candidates["D50"]) - np.log(median_d50))
        candidates = candidates.sort_values(
            ["_distance", "_d50_distance", "num_dp", "sample_ID"],
            ascending=[True, True, False, True],
        )
        row = candidates.iloc[0].copy()
        row["target_name"] = target_name
        row["target_label"] = label
        row["target_skew"] = target_value
        row["target_quantile"] = quantile if quantile is not None else np.nan
        selected.append(row)
    return selected, thresholds


def observed_density_steps(
    row: pd.Series,
    size_lookup: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = size_lookup.to_numpy(dtype=float)
    values = pd.to_numeric(row[size_lookup.index], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(sizes) & np.isfinite(values)
    sizes = sizes[valid]
    values = np.clip(values[valid], 0.0, 100.0)
    order = np.argsort(sizes)
    sizes = sizes[order]
    values = values[order]

    unique_sizes, inverse = np.unique(sizes, return_inverse=True)
    unique_values = np.array(
        [np.nanmean(values[inverse == i]) for i in range(len(unique_sizes))]
    )
    unique_values = np.maximum.accumulate(unique_values)
    unique_values = np.clip(unique_values, 0.0, 100.0)

    if len(unique_sizes) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    widths = np.diff(np.log2(unique_sizes))
    mass = np.diff(unique_values)
    valid_intervals = widths > 0
    density = np.full_like(mass, np.nan, dtype=float)
    density[valid_intervals] = mass[valid_intervals] / widths[valid_intervals]
    density = np.clip(density, 0.0, np.inf)
    return unique_sizes, density


def format_relative_ticks(ticks: np.ndarray) -> list[str]:
    labels = []
    for tick in ticks:
        if tick < 1:
            labels.append(f"1/{int(round(1 / tick))}")
        else:
            labels.append(f"{int(round(tick))}")
    return labels


def fit_row_for_sample(fit: pd.DataFrame, sample: pd.Series) -> pd.Series:
    match = fit[
        (fit["site_no"].astype("string") == str(sample["site_no"]))
        & (fit["sample_ID"] == sample["sample_ID"])
    ]
    if match.empty:
        raise ValueError(
            f"No fit found for site_no={sample['site_no']}, sample_ID={sample['sample_ID']}"
        )
    return match.iloc[0]


def write_summary(
    path: Path,
    selected: list[pd.Series],
    fits: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
) -> None:
    rows = []
    for sample in selected:
        row = {
            "target_name": sample["target_name"],
            "target_label": sample["target_label"],
            "target_skew": sample["target_skew"],
            "site_no": sample["site_no"],
            "sample_ID": int(sample["sample_ID"]),
            "skew_FW": sample["skew_FW"],
            "D50": sample["D50"],
            "num_dp": sample["num_dp"],
            "best_function(s)": sample.get("best_function(s)", ""),
            "skew_p10_threshold": thresholds["P10"],
            "skew_p90_threshold": thresholds["P90"],
        }
        for function_name, fit in fits.items():
            fit_row = fit_row_for_sample(fit, sample)
            row[f"{function_name}_RMSE"] = fit_row.get("RMSE", np.nan)
            row[f"{function_name}_BIC"] = fit_row.get("BIC", np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_single_samples(
    selected: list[pd.Series],
    fits: dict[str, pd.DataFrame],
    size_lookup: pd.Series,
    out_prefix: Path,
    n_grid: int,
) -> None:
    setup_style()
    x_grid = 2.0 ** np.linspace(np.log2(FULL_XLIM[0]), np.log2(FULL_XLIM[1]), n_grid)
    fig, axes = plt.subplots(2, 3, figsize=(6.0, 4.0), sharex=False, sharey=False)
    top_axes = axes[0]
    bottom_axes = axes[1]
    top_letters = ["a", "b", "c"]
    bottom_letters = ["d", "e", "f"]

    for index, (ax, sample, letter) in enumerate(zip(top_axes, selected, top_letters)):
        panel_max_y = 0.0
        edges, observed_density = observed_density_steps(sample, size_lookup)
        if len(edges) > 1:
            ax.stairs(
                observed_density,
                edges,
                baseline=0,
                fill=True,
                facecolor=OBS_FACE,
                edgecolor=OBS_EDGE,
                linewidth=0.85,
                alpha=0.85,
            )
            panel_max_y = max(panel_max_y, np.nanmax(observed_density))

        for function_name in FUNCTION_ORDER:
            fit_row = fit_row_for_sample(fits[function_name], sample)
            density = eval_density(function_name, x_grid, fit_row)
            density = np.asarray(density, dtype=float)
            density[~np.isfinite(density)] = np.nan
            density = np.clip(density, 0.0, np.inf)
            panel_max_y = max(panel_max_y, np.nanmax(density))
            ax.plot(
                x_grid,
                density,
                color=FUNCTION_COLORS[function_name],
                linestyle=FUNCTION_STYLES[function_name],
                linewidth=1.25,
            )

        style_axis(ax)
        ax.text(
            -0.02,
            1.035,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            clip_on=False,
        )
        ax.set_xscale("log", base=2)
        if index == 1:
            ax.set_xlim(*MID_XLIM)
            ax.set_xticks(MID_XTICKS)
            ax.set_xticklabels(format_relative_ticks(MID_XTICKS))
        else:
            ax.set_xlim(*FULL_XLIM)
            ax.set_xticks(FULL_XTICKS)
            ax.set_xticklabels(format_relative_ticks(FULL_XTICKS))
        ax.tick_params(colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
        if np.isfinite(panel_max_y) and panel_max_y > 0:
            ax.set_ylim(0, panel_max_y * 1.12)

    for ax, function_name, letter in zip(bottom_axes, BOTTOM_FUNCTION_ORDER, bottom_letters):
        panel_max_y = 0.0
        for sample in selected:
            fit_row = fit_row_for_sample(fits[function_name], sample)
            density = eval_density(function_name, x_grid, fit_row)
            density = np.asarray(density, dtype=float)
            density[~np.isfinite(density)] = np.nan
            density = np.clip(density, 0.0, np.inf)
            panel_max_y = max(panel_max_y, np.nanmax(density))
            ax.plot(
                x_grid,
                density,
                color=FUNCTION_COLORS[function_name],
                linestyle=SAMPLE_STYLES[sample["target_name"]],
                linewidth=1.15,
            )

        style_axis(ax)
        ax.text(
            -0.02,
            1.035,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            clip_on=False,
        )
        ax.set_xscale("log", base=2)
        ax.set_xlim(*FULL_XLIM)
        ax.set_xticks(FULL_XTICKS)
        ax.set_xticklabels(format_relative_ticks(FULL_XTICKS))
        ax.tick_params(colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
        if np.isfinite(panel_max_y) and panel_max_y > 0:
            ax.set_ylim(0, panel_max_y * 1.12)

    fig.text(
        0.035,
        0.55,
        "Probability density (% per log2 mm)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8,
    )
    fig.text(
        0.55,
        0.135,
        "Particle size (mm, log2 scale)",
        ha="center",
        va="center",
        fontsize=8,
    )

    handles = [
        Patch(facecolor=OBS_FACE, edgecolor=OBS_EDGE, label="observed data"),
        Line2D(
            [0],
            [0],
            color=FUNCTION_COLORS["GIG_2p"],
            linestyle=FUNCTION_STYLES["GIG_2p"],
            linewidth=1.25,
            label=FUNCTION_LABELS["GIG_2p"],
        ),
        Line2D(
            [0],
            [0],
            color=FUNCTION_COLORS["Lognormal"],
            linestyle=FUNCTION_STYLES["Lognormal"],
            linewidth=1.25,
            label=FUNCTION_LABELS["Lognormal"],
        ),
        Line2D(
            [0],
            [0],
            color=FUNCTION_COLORS["Weibull"],
            linestyle=FUNCTION_STYLES["Weibull"],
            linewidth=1.25,
            label=FUNCTION_LABELS["Weibull"],
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=2.5,
        columnspacing=1.35,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.965,
        bottom=0.20,
        wspace=0.26,
        hspace=0.38,
    )
    fig.savefig(out_prefix.with_suffix(".png"), dpi=600)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select individual samples closest to P10, zero, and P90 skewness, "
            "then plot observed PSD densities and fitted GIG/lognormal/Weibull curves."
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--sizes", type=Path, default=SIZE_PATH)
    parser.add_argument("--out-prefix", type=Path, default=SAMPLE_OUT_PREFIX)
    parser.add_argument("--n-grid", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size_lookup = load_size_lookup(args.sizes)
    data = load_data(args.data, size_lookup)
    fits = load_fits()
    data = samples_with_all_fits(data, fits)
    selected, thresholds = select_representatives(data)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_single_samples(selected, fits, size_lookup, args.out_prefix, args.n_grid)


if __name__ == "__main__":
    main()
