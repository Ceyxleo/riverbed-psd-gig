#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "tables" / "bic_matrix.tsv"
DEFAULT_OUTPUT_PREFIX = ROOT / "figures" / "Figure_4"

GROUP_COLORS = {
    "2-parameter": "#5B6F8E",
    "3-parameter": "#B75D5A",
}
TOP_FUNCTION_COLORS = {
    "GIG_3p": "#B75D5A",
    "GIG_2p": "#5B6F8E",
    "Logn_PL": "#D7BA6A",
    "Lognormal": "#009E73",
    "GLH": "#5BA6A0",
    "Weibull": "#56B4E9",
    "Rest": "#C9C9C6",
}
FALLBACK_TOP_COLORS = ["#B75D5A", "#5B6F8E", "#D7BA6A", "#009E73", "#5BA6A0", "#56B4E9"]

HIGHLIGHT_EDGE = "#202020"
GRID_COLOR = "#E6E2DB"
TEXT_COLOR = "#222222"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a manuscript-style model performance figure: a full-width "
            "ranked raw-metric summary, top-function violin+box distributions, "
            "and a top-functions-plus-rest donut chart."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metric", default="BIC")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--sep", default="auto", help="'auto', 'tab', 'comma', or a literal delimiter.")
    parser.add_argument(
        "--id-columns",
        nargs="+",
        default=["sample_ID", "site_no"],
        help="Columns excluded before treating remaining columns as functions.",
    )
    parser.add_argument("--n-two-parameter", type=int, default=15)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--highlight", nargs="*", default=["GIG_2p", "GIG_3p"])
    parser.add_argument(
        "--distribution-values",
        choices=["delta", "raw"],
        default="raw",
        help="Use sample-normalized delta values or raw metric values in panel b.",
    )
    parser.add_argument(
        "--distribution-x-percentile",
        type=float,
        default=99.0,
        help="Right x-axis window for panel b; use 100 for full range.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Use higher metric values as better, e.g. for R2. BIC/RMSE should leave this off.",
    )
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--formats", nargs="+", default=["png"])
    return parser.parse_args()


def delimiter_from_arg(path: Path, sep: str) -> str:
    if sep == "auto":
        return "," if path.suffix.lower() == ".csv" else "\t"
    if sep == "tab":
        return "\t"
    if sep == "comma":
        return ","
    return sep


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def load_metric_table(path: Path, sep: str, id_columns: list[str]) -> tuple[pd.DataFrame, list[str], int]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path, sep=sep)
    function_columns = [col for col in data.columns if col not in id_columns]
    if not function_columns:
        raise ValueError("No function columns found after removing id columns.")

    metrics = data[function_columns].apply(pd.to_numeric, errors="coerce")
    nonfinite_count = int(np.isinf(metrics.to_numpy(dtype=float)).sum())
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    usable_rows = metrics.notna().any(axis=1)
    metrics = metrics.loc[usable_rows].copy()
    return metrics, function_columns, nonfinite_count


def group_for_columns(columns: list[str], n_two_parameter: int) -> dict[str, str]:
    return {
        function: "2-parameter" if idx < n_two_parameter else "3-parameter"
        for idx, function in enumerate(columns)
    }


def compute_delta(metrics: pd.DataFrame, lower_is_better: bool) -> pd.DataFrame:
    if lower_is_better:
        best_by_sample = metrics.min(axis=1, skipna=True)
        return metrics.sub(best_by_sample, axis=0)
    best_by_sample = metrics.max(axis=1, skipna=True)
    return best_by_sample.sub(metrics, axis=0)


def percentile(values: pd.Series, q: float) -> float:
    return float(np.nanpercentile(values.to_numpy(dtype=float), q))


def build_summary(
    metrics: pd.DataFrame,
    function_columns: list[str],
    groups: dict[str, str],
    tie_tolerance: float,
    lower_is_better: bool,
) -> pd.DataFrame:
    rows = []
    n_samples = len(metrics)
    if lower_is_better:
        best_by_sample = metrics.min(axis=1, skipna=True)
        winners = metrics.le(best_by_sample.add(tie_tolerance), axis=0)
    else:
        best_by_sample = metrics.max(axis=1, skipna=True)
        winners = metrics.ge(best_by_sample.sub(tie_tolerance), axis=0)

    for idx, function in enumerate(function_columns):
        metric_values = metrics[function]
        best_count = int(winners[function].sum())
        rows.append(
            {
                "function": function,
                "function_order": idx + 1,
                "parameter_group": groups[function],
                "n_valid": int(metric_values.notna().sum()),
                "n_missing": int(metric_values.isna().sum()),
                "median_metric": float(metric_values.median(skipna=True)),
                "metric_q10": percentile(metric_values, 10),
                "metric_q25": percentile(metric_values, 25),
                "metric_q75": percentile(metric_values, 75),
                "metric_q90": percentile(metric_values, 90),
                "best_count": best_count,
                "best_pct": best_count / n_samples * 100,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["median_metric", "function_order"], ascending=[lower_is_better, True])
        .reset_index(drop=True)
    )


def metric_label(metric: str) -> str:
    return rf"$\Delta${metric.strip()}"


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors=TEXT_COLOR)


def set_bold_labels(labels, highlights: set[str]) -> None:
    for tick in labels:
        if tick.get_text() in highlights:
            tick.set_fontweight("bold")
            tick.set_color("#111111")


def expand_limit(max_value: float) -> tuple[float, float]:
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0
    return 0, max_value * 1.08


def padded_limit(low: float, high: float, pad_fraction: float = 0.08) -> tuple[float, float]:
    if not np.isfinite(low) or not np.isfinite(high):
        return 0, 1
    if low == high:
        return low - 0.5, high + 0.5
    span = high - low
    return low - span * pad_fraction, high + span * pad_fraction


def robust_xlim(values: np.ndarray, right_percentile: float, pad_fraction: float = 0.06) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, 1
    if right_percentile >= 100:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    else:
        low = float(np.nanpercentile(finite, max(0, 100 - right_percentile)))
        high = float(np.nanpercentile(finite, right_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if low == high:
        low -= 0.5
        high += 0.5
    span = high - low
    return low - span * pad_fraction, high + span * pad_fraction


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def plot_ranked_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    groups: dict[str, str],
    highlights: set[str],
    metric: str,
    top_n: int,
) -> None:
    x = np.arange(len(summary))
    ax.axvspan(-0.5, top_n - 0.5, color="#F3E8D8", alpha=0.52, zorder=0)
    ax.axvspan(top_n - 0.5, len(summary) - 0.5, color="#EEF2F4", alpha=0.72, zorder=0)
    ax.vlines(x, summary["metric_q10"], summary["metric_q90"], color="#B8B4AE", linewidth=0.7, zorder=1)

    for pos, row in summary.iterrows():
        function = row["function"]
        color = GROUP_COLORS[groups[function]]
        is_highlight = function in highlights
        ax.vlines(
            pos,
            row["metric_q25"],
            row["metric_q75"],
            color=color,
            linewidth=2.2 if is_highlight else 1.7,
            alpha=0.95,
            zorder=2,
        )
        ax.scatter(
            pos,
            row["median_metric"],
            s=34 if is_highlight else 19,
            facecolor=color,
            edgecolor=HIGHLIGHT_EDGE if is_highlight else "white",
            linewidth=0.75,
            zorder=3,
        )
        ax.annotate(
            f"{row['median_metric']:.1f}",
            xy=(pos, row["median_metric"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=4.9,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["function"], rotation=62, ha="right", rotation_mode="anchor")
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    set_bold_labels(ax.get_xticklabels(), highlights)
    style_axis(ax, grid_axis="y")
    ax.set_xlim(-0.75, len(summary) - 0.25)
    ax.set_ylim(
        padded_limit(
            float(summary["metric_q10"].min()),
            float(summary["metric_q90"].max()),
        )
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_COLORS["2-parameter"],
            markeredgecolor="white",
            markersize=5,
            label="2-parameter",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_COLORS["3-parameter"],
            markeredgecolor="white",
            markersize=5,
            label="3-parameter",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        ncol=1,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


def plot_top_distributions(
    ax: plt.Axes,
    values: pd.DataFrame,
    top_functions: list[str],
    groups: dict[str, str],
    highlights: set[str],
    metric: str,
    use_delta_values: bool,
    x_percentile: float,
) -> None:
    arrays = [values[function].dropna().to_numpy(dtype=float) for function in top_functions]
    positions = np.arange(len(top_functions))

    parts = ax.violinplot(
        arrays,
        positions=positions,
        vert=False,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, function in zip(parts["bodies"], top_functions):
        body.set_facecolor(GROUP_COLORS[groups[function]])
        body.set_edgecolor("none")
        body.set_alpha(0.58 if function in highlights else 0.36)

    box = ax.boxplot(
        arrays,
        positions=positions,
        vert=False,
        widths=0.23,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "#111111", "linewidth": 1.15},
        whiskerprops={"color": "#111111", "linewidth": 0.7},
        capprops={"color": "#111111", "linewidth": 0.7},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("#111111")
        patch.set_alpha(0.88)
        patch.set_linewidth(0.75)

    for pos, function, arr in zip(positions, top_functions, arrays):
        median = float(np.nanmedian(arr))
        ax.scatter(
            median,
            pos,
            s=24 if function in highlights else 17,
            facecolor=GROUP_COLORS[groups[function]],
            edgecolor=HIGHLIGHT_EDGE,
            linewidth=0.65,
            zorder=4,
        )
        ax.annotate(
            f"{median:.1f}",
            xy=(median, pos),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.9,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=5,
        )

    combined = np.concatenate([arr for arr in arrays if len(arr) > 0])
    xmin, xmax = robust_xlim(combined, x_percentile, pad_fraction=0.05)
    window_label = "" if x_percentile >= 100 else f" ({x_percentile:g}th percentile window)"

    label = metric_label(metric) if use_delta_values else metric
    ax.set_yticks(positions)
    ax.set_yticklabels(top_functions)
    ax.invert_yaxis()
    ax.set_xlabel(f"{label}{window_label}")
    set_bold_labels(ax.get_yticklabels(), highlights)
    style_axis(ax, grid_axis="x")
    ax.set_xlim((0, xmax) if use_delta_values else (xmin, xmax))


def donut_colors(top_functions: list[str]) -> dict[str, str]:
    colors = {}
    for idx, function in enumerate(top_functions):
        colors[function] = TOP_FUNCTION_COLORS.get(function, FALLBACK_TOP_COLORS[idx % len(FALLBACK_TOP_COLORS)])
    colors["Rest"] = TOP_FUNCTION_COLORS["Rest"]
    return colors


def plot_best_donut(
    ax: plt.Axes,
    summary: pd.DataFrame,
    top_functions: list[str],
    n_samples: int,
    metric: str,
) -> pd.DataFrame:
    rows = []
    for function in top_functions:
        pct = float(summary.loc[summary["function"] == function, "best_pct"].iloc[0])
        rows.append({"function": function, "best_pct": pct})
    rows.append({"function": "Rest", "best_pct": max(0.0, 100.0 - sum(row["best_pct"] for row in rows))})
    share = pd.DataFrame(rows)

    colors = donut_colors(top_functions)
    radius = 1.25
    wedges, _ = ax.pie(
        share["best_pct"],
        startangle=90,
        counterclock=False,
        radius=radius,
        colors=[colors[function] for function in share["function"]],
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 0.7},
    )
    ax.text(0, 0.05, f"n = {n_samples:,}", ha="center", va="center", fontsize=7.2, color=TEXT_COLOR)
    ax.text(0, -0.15, "samples", ha="center", va="center", fontsize=6.1, color="#555555")

    label_specs = []
    for wedge, row in zip(wedges, share.itertuples(index=False)):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        side = 1 if np.cos(angle) >= 0 else -1
        xy = (np.cos(angle) * radius * 0.86, np.sin(angle) * radius * 0.86)
        text_xy = (side * radius * 1.22, np.sin(angle) * radius * 1.08)
        label_specs.append([row.function, row.best_pct, xy, text_xy, side])

    for side in [-1, 1]:
        side_specs = [spec for spec in label_specs if spec[4] == side]
        side_specs.sort(key=lambda spec: spec[3][1])
        min_gap = 0.18
        for idx in range(1, len(side_specs)):
            if side_specs[idx][3][1] - side_specs[idx - 1][3][1] < min_gap:
                side_specs[idx][3] = (
                    side_specs[idx][3][0],
                    side_specs[idx - 1][3][1] + min_gap,
                )

    for function, pct, xy, text_xy, side in label_specs:
        ax.annotate(
            f"{pct:.1f}%",
            xy=xy,
            xytext=text_xy,
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=5.8,
            color=TEXT_COLOR,
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.45,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    legend_labels = [function for function in top_functions]
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=colors[function],
            markeredgecolor="white",
            markersize=5,
            label=label,
        )
        for function, label in zip(top_functions, legend_labels)
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.08),
        ncol=3,
        handletextpad=0.35,
        borderaxespad=0,
        columnspacing=0.75,
        labelspacing=0.42,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.52, 1.48)
    return share


def make_figure(
    metrics: pd.DataFrame,
    delta: pd.DataFrame,
    summary: pd.DataFrame,
    groups: dict[str, str],
    metric: str,
    top_n: int,
    highlights: set[str],
    distribution_values: str,
    distribution_x_percentile: float,
) -> tuple[plt.Figure, pd.DataFrame]:
    configure_matplotlib()
    top_functions = summary.head(top_n)["function"].tolist()
    distribution_source = delta if distribution_values == "delta" else metrics

    fig = plt.figure(figsize=(7.25, 6.35))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1.4, 1.0],
        width_ratios=[1, 1, 1],
        hspace=0.3,
        wspace=0.2,
    )
    ax_rank = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, :2])
    ax_donut = fig.add_subplot(gs[1, 2])

    plot_ranked_metric(ax_rank, summary, groups, highlights, metric, top_n)
    plot_top_distributions(
        ax_dist,
        distribution_source,
        top_functions,
        groups,
        highlights,
        metric,
        use_delta_values=distribution_values == "delta",
        x_percentile=distribution_x_percentile,
    )
    share = plot_best_donut(ax_donut, summary, top_functions, len(metrics), metric)

    panel_label(ax_rank, "a", x=-0.035, y=1.03)
    panel_label(ax_dist, "b", x=-0.055)
    panel_label(ax_donut, "c", x=-0.10)
    return fig, share


def main() -> None:
    args = parse_args()
    lower_is_better = not args.higher_is_better
    sep = delimiter_from_arg(args.input, args.sep)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics, function_columns, nonfinite_count = load_metric_table(args.input, sep, args.id_columns)
    groups = group_for_columns(function_columns, args.n_two_parameter)
    delta = compute_delta(metrics, lower_is_better=lower_is_better)
    summary = build_summary(metrics, function_columns, groups, args.tie_tolerance, lower_is_better)

    fig, share = make_figure(
        metrics,
        delta,
        summary,
        groups,
        args.metric,
        args.top_n,
        set(args.highlight),
        args.distribution_values,
        args.distribution_x_percentile,
    )

    saved = []
    for fmt in args.formats:
        fmt_clean = fmt.lower().lstrip(".")
        out = args.output_prefix.with_suffix(f".{fmt_clean}")
        fig.savefig(out, dpi=600 if fmt_clean in {"png", "tif", "tiff"} else None)
        saved.append(out)
    plt.close(fig)

    print(f"Input: {args.input}")
    print(f"Samples used: {len(metrics):,}")
    if nonfinite_count:
        print(f"Non-finite metric values treated as missing: {nonfinite_count:,}")
    print(f"Functions: {len(function_columns)}")
    print(f"Top functions by median {args.metric}:")
    for row in summary.head(args.top_n).itertuples(index=False):
        print(f"  {row.function}: median_{args.metric}={row.median_metric:.3f}, best_pct={row.best_pct:.1f}%")
    print("Best-function share:")
    for row in share.itertuples(index=False):
        print(f"  {row.function}: {row.best_pct:.1f}%")
    for out in saved:
        print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
