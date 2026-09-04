#!/usr/bin/env python3
"""Redesign Supplementary Figure S3/S4 BIC-difference histograms."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsave import add_format_argument, save_figure  # noqa: E402

DEFAULT_BIC_PATH = ROOT / "outputs" / "tables" / "bic_matrix.tsv"
DEFAULT_OUT_PREFIX = ROOT / "figures" / "Figure_S3"
DEFAULT_SUMMARY_PATH = ROOT / "outputs" / "tables" / "figure_S3_summary.csv"
MPLCONFIG_DIR = ROOT / "outputs" / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

BINS = [
    ("<-10", float("-inf"), -10.0, -12.0, -10.0),
    ("-10 to -8", -10.0, -8.0, -10.0, -8.0),
    ("-8 to -6", -8.0, -6.0, -8.0, -6.0),
    ("-6 to -4", -6.0, -4.0, -6.0, -4.0),
    ("-4 to -2", -4.0, -2.0, -4.0, -2.0),
    ("-2 to 0", -2.0, 0.0, -2.0, 0.0),
    ("0 to 2", 0.0, 2.0, 0.0, 2.0),
    ("2 to 4", 2.0, 4.0, 2.0, 4.0),
    ("4 to 6", 4.0, 6.0, 4.0, 6.0),
    ("6 to 8", 6.0, 8.0, 6.0, 8.0),
    ("8 to 10", 8.0, 10.0, 8.0, 10.0),
    (">10", 10.0, float("inf"), 10.0, 12.0),
]

DISPLAY_EDGES = [BINS[0][3]] + [bin_spec[4] for bin_spec in BINS]
ZERO_INDEX = 6

PANELS = [
    ("a", "GIG_2p", "Lognormal"),
    ("b", "GIG_2p", "Weibull"),
    ("c", "Lognormal", "Weibull"),
    ("d", "GIG_3p", "Logn_PL"),
    ("e", "GIG_3p", "GLH"),
    ("f", "Logn_PL", "GLH"),
    ("g", "GIG_3p", "GIG_2p"),
    ("h", "Logn_PL", "Lognormal"),
    ("i", "GLH", "Weibull"),
]

NEG_FILL = "#DCEAF7"
POS_FILL = "#F3C6A5"
LINE_COLOR = "#1F3F5B"
ZERO_COLOR = "#4A4A4A"
TEXT_COLOR = "#1F1F1F"
GRID_COLOR = "#E6E6E6"
PERCENT_POS_BY_ROW = {0: (-4, 1000), 1: (-2, 1000), 2: (-8, 2000)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready BIC-difference histogram panels."
    )
    parser.add_argument("--bic-path", type=Path, default=DEFAULT_BIC_PATH)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    add_format_argument(parser)
    return parser.parse_args()


def setup_style() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def count_bins(diff):
    import numpy as np

    counts = []
    for _, low, high, _, _ in BINS:
        if np.isneginf(low):
            mask = diff < high
        elif np.isposinf(high):
            mask = diff >= low
        else:
            mask = (diff >= low) & (diff < high)
        counts.append(int(mask.sum()))
    counts = np.array(counts, dtype=int)
    return counts


def collect_panel_data(bic_path: Path):
    import pandas as pd

    data = pd.read_csv(bic_path, sep="\t", dtype={"site_no": "string"})
    required = {name for _, left, right in PANELS for name in (left, right)}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing BIC columns: {sorted(missing)}")

    rows = []
    panel_data = []
    for label, left, right in PANELS:
        diff = data[left] - data[right]
        counts = count_bins(diff)
        total = int(diff.notna().sum())
        n_gt0 = int((diff > 0).sum())
        n_lt0 = int((diff < 0).sum())
        n_eq0 = int((diff == 0).sum())
        panel_data.append(
            {
                "panel": label,
                "left": left,
                "right": right,
                "counts": counts,
                "total": total,
                "pct_gt0": n_gt0 / total * 100.0,
                "pct_lt0": n_lt0 / total * 100.0,
            }
        )
        for bin_label, count in zip([item[0] for item in BINS], counts):
            rows.append(
                {
                    "panel": label,
                    "comparison": f"{left} - {right}",
                    "bin": bin_label,
                    "count": int(count),
                    "n_total": total,
                    "n_lt0": n_lt0,
                    "n_gt0": n_gt0,
                    "n_eq0": n_eq0,
                    "pct_lt0": n_lt0 / total * 100.0,
                    "pct_gt0": n_gt0 / total * 100.0,
                }
            )

    return panel_data, pd.DataFrame(rows)


def nice_upper(value: float) -> int:
    import math

    if value <= 0:
        return 1
    step = 1000 if value > 7000 else 500
    return int(math.ceil(value * 1.18 / step) * step)


def style_axis(ax, row: int, col: int, ylim: int) -> None:
    ax.set_xlim(DISPLAY_EDGES[0], DISPLAY_EDGES[-1])
    ax.set_ylim(0, ylim)
    xticks = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    xtick_labels = ["-10", "-8", "-6", "-4", "-2", "0", "2", "4", "6", "8", "10"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555", labelcolor="#555555", pad=1.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#777777")
    if col == 0:
        ax.set_ylabel("Number of samples")
    else:
        ax.set_yticklabels([])
    if row != 2:
        ax.set_xticklabels([])
    else:
        tail_label_kwargs = {
            "transform": ax.get_xaxis_transform(),
            "ha": "center",
            "va": "bottom",
            "fontsize": 5.6,
            "color": "#555555",
        }
        ax.text(-11, 0.025, "<-10", **tail_label_kwargs)
        ax.text(11, 0.025, ">10", **tail_label_kwargs)


def plot_panel(ax, item, row: int) -> None:
    import numpy as np

    edges = np.array(DISPLAY_EDGES, dtype=float)
    counts = item["counts"]

    ax.stairs(counts[:ZERO_INDEX], edges[: ZERO_INDEX + 1], baseline=0, fill=True, color=NEG_FILL, zorder=2)
    ax.stairs(counts[ZERO_INDEX:], edges[ZERO_INDEX:], baseline=0, fill=True, color=POS_FILL, zorder=2)
    ax.stairs(counts, edges, baseline=0, fill=False, color=LINE_COLOR, linewidth=1.35, zorder=4)
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.9, linestyle=(0, (3, 2)), zorder=5)

    ax.set_title(
        f"{item['panel']}  {item['left']} - {item['right']}",
        loc="left",
        pad=2.5,
        color=TEXT_COLOR,
        fontweight="bold",
    )

    ax.text(
        PERCENT_POS_BY_ROW[row][0],
        PERCENT_POS_BY_ROW[row][1],
        f"{item['pct_lt0']:.1f}%",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#1F3F5B",
        linespacing=0.9,
        zorder=6,
    )


def make_figure(panel_data, out_prefix: Path, fmt: str = "png") -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    setup_style()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    row_limits = []
    for row in range(3):
        row_items = panel_data[row * 3 : (row + 1) * 3]
        row_limits.append(nice_upper(max(item["counts"].max() for item in row_items)))

    fig, axes = plt.subplots(3, 3, figsize=(7.15, 6.15), sharex=True)

    for index, item in enumerate(panel_data):
        row, col = divmod(index, 3)
        ax = axes[row, col]
        style_axis(ax, row, col, row_limits[row])
        plot_panel(ax, item, row)

    handles = [
        Patch(facecolor=NEG_FILL, edgecolor=LINE_COLOR, linewidth=0.8, label="first model lower BIC"),
        Patch(facecolor=POS_FILL, edgecolor=LINE_COLOR, linewidth=0.8, label="second model lower BIC"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.0),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=2.0,
    )
    fig.supxlabel(
        "Difference = BIC(first model) - BIC(second model)",
        y=0.015,
        fontsize=7,
        color="#333333",
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.085, top=0.93, wspace=0.12, hspace=0.24)

    png_path = out_prefix.with_suffix(".png")
    save_figure(fig, png_path.with_suffix(""), fmt, dpi=600)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    panel_data, summary = collect_panel_data(args.bic_path)
    png_path = make_figure(panel_data, args.out_prefix, args.format)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_path, index=False)
    print(f"Wrote {args.summary_path}")


if __name__ == "__main__":
    main()
