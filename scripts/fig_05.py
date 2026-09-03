from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "outputs" / "water" / "sample_water_metrics.csv"
DEFAULT_OUT = ROOT / "figures" / "Figure_5"

FUNCTION_ORDER = ["GIG_2p", "Lognormal", "Weibull"]
FUNCTION_LABELS = {"GIG_2p": "GIG", "Lognormal": "Lognormal", "Weibull": "Weibull"}

D_METRICS = [
    ("D16", "D16", "D16_reference", {fn: f"D16_{fn}" for fn in FUNCTION_ORDER}),
    ("D50", "D50", "D50_reference", {fn: f"D50_{fn}" for fn in FUNCTION_ORDER}),
    ("D84", "D84", "D84_reference", {fn: f"D84_{fn}" for fn in FUNCTION_ORDER}),
]

WATER_METRICS = [
    (
        "bedload",
        "Bedload\ntransport",
        "bedload_transport_reference_m2_s",
        {fn: f"bedload_transport_{fn}_m2_s" for fn in FUNCTION_ORDER},
        True,
    ),
    (
        "critical_shear",
        "Critical shear\nstress",
        "critical_shear_stress_reference_Pa",
        {fn: f"critical_shear_stress_{fn}_Pa" for fn in FUNCTION_ORDER},
        False,
    ),
    (
        "flood_stage",
        "100-year\nflood stage",
        "flood_stage_reference_m",
        {fn: f"flood_stage_{fn}_m" for fn in FUNCTION_ORDER},
        False,
    ),
    (
        "fredle_index",
        "Fredle\nIndex",
        "fredle_index_reference",
        {fn: f"fredle_index_{fn}" for fn in FUNCTION_ORDER},
        False,
    ),
]

TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"
GRID_COLOR = "#FFFFFF"
GROUP_LINE = "#77736C"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.55,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "savefig.pad_inches": 0.035,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def error_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "clean_green_error",
        ["#6FAE9B", "#93C6B4", "#B9DCCF", "#D9ECE4", "#EEF6F1", "#FBFCF7"],
    )


def relative_error(reference: pd.Series, prediction: pd.Series) -> pd.Series:
    valid = reference.notna() & prediction.notna() & (reference != 0)
    err = pd.Series(np.nan, index=reference.index, dtype=float)
    err.loc[valid] = (prediction.loc[valid] - reference.loc[valid]) / reference.loc[valid] * 100.0
    return err


def summarize_errors(errors: pd.Series) -> dict[str, float]:
    clean = errors.replace([np.inf, -np.inf], np.nan).dropna()
    abs_clean = clean.abs()
    return {
        "n": int(len(clean)),
        "mean_relative_error": float(clean.mean()),
        "median_absolute_relative_error": float(abs_clean.median()),
        "rmsre": float(np.sqrt(np.mean(np.square(clean))) if len(clean) else np.nan),
    }


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric_key, metric_label, ref_col, pred_cols in D_METRICS:
        subset = data.dropna(subset=[ref_col] + [pred_cols[fn] for fn in FUNCTION_ORDER]).copy()
        subset = subset.loc[subset[ref_col] != 0].copy()
        for fn in FUNCTION_ORDER:
            rows.append(
                {
                    "group": "grain_size",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "function": fn,
                    "function_label": FUNCTION_LABELS[fn],
                    "n_sites": int(subset["site_no"].nunique()),
                    **summarize_errors(relative_error(subset[ref_col], subset[pred_cols[fn]])),
                }
            )

    for metric_key, metric_label, ref_col, pred_cols, drop_min_reference in WATER_METRICS:
        subset = data.dropna(subset=[ref_col] + [pred_cols[fn] for fn in FUNCTION_ORDER]).copy()
        if drop_min_reference and not subset.empty:
            # Match the original Figure 5 notebook's exclusion of the minimum bedload reference value.
            subset = subset.loc[subset[ref_col] != subset[ref_col].min()].copy()
        else:
            subset = subset.loc[subset[ref_col] != 0].copy()
        for fn in FUNCTION_ORDER:
            rows.append(
                {
                    "group": "water_security",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "function": fn,
                    "function_label": FUNCTION_LABELS[fn],
                    "n_sites": int(subset["site_no"].nunique()),
                    **summarize_errors(relative_error(subset[ref_col], subset[pred_cols[fn]])),
                }
            )

    return pd.DataFrame(rows)


def annotate_cells(ax: plt.Axes, matrix: np.ndarray) -> None:
    column_minima = np.nanmin(matrix, axis=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            is_column_minimum = np.isclose(value, column_minima[j], rtol=0, atol=1e-10)
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.3,
                color=TEXT_COLOR,
                fontweight="bold" if is_column_minimum else "normal",
            )


def make_figure(metrics_path: Path, out_prefix: Path) -> None:
    setup_style()
    data = pd.read_csv(metrics_path, dtype={"site_no": str})
    summary = build_summary(data)

    column_specs = [(item[0], item[1]) for item in D_METRICS] + [
        (item[0], item[1]) for item in WATER_METRICS
    ]
    matrix = np.full((len(FUNCTION_ORDER), len(column_specs)), np.nan)
    for i, fn in enumerate(FUNCTION_ORDER):
        for j, (metric_key, _) in enumerate(column_specs):
            sub = summary[(summary["function"] == fn) & (summary["metric"] == metric_key)]
            if not sub.empty:
                matrix[i, j] = sub["rmsre"].iloc[0]

    vmax = float(np.ceil(np.nanmax(matrix) / 5.0) * 5.0)
    vmax = max(30.0, vmax)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = error_cmap()

    fig = plt.figure(figsize=(5.75, 3.45))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.035], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    annotate_cells(ax, matrix)

    ax.set_xticks(np.arange(len(column_specs)))
    ax.set_yticks(np.arange(len(FUNCTION_ORDER)))
    ax.set_xticklabels([label for _, label in column_specs])
    ax.set_yticklabels([FUNCTION_LABELS[fn] for fn in FUNCTION_ORDER])
    ax.set_xticks(np.arange(-0.5, len(column_specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FUNCTION_ORDER), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", length=0, pad=8, colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
    ax.tick_params(axis="y", length=0, pad=8, colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.plot(
        [2.5, 2.5],
        [-0.5, len(FUNCTION_ORDER) - 0.5],
        color=GROUP_LINE,
        linewidth=1.8,
        solid_capstyle="butt",
        clip_on=True,
        zorder=4,
    )
    ax.text(
        1.0,
        1.055,
        "Grain-size percentiles",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        4.5,
        1.055,
        "Water-security variables",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        clip_on=False,
    )

    ax.set_ylim(len(FUNCTION_ORDER) - 0.5, -0.5)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("RMSRE(%)", rotation=270, labelpad=11)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=5.8, colors=AXIS_COLOR)

    fig.subplots_adjust(left=0.12, right=0.87, top=0.80, bottom=0.18, wspace=0.06)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_prefix.with_suffix(".png"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)

    print(out_prefix.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Single heatmap Figure 5 redesign.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    make_figure(args.metrics.expanduser(), args.out.expanduser())


if __name__ == "__main__":
    main()
