#!/usr/bin/env python3
"""Supplementary Fig. S5 -- GIG, lognormal and Weibull on the Lower Rhine.

(a) BIC across the 67 Lower Rhine samples, median printed under each violin.
(b) the same for RMSE.
(c) best-fit composition within each Folk-Ward skewness class, where the best-fit
    set holds every function within 2 BIC units of the sample minimum.

Run `python scripts/06_rhine_summary.py` first; this reads its outputs.

    python scripts/fig_S05.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",
                      str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from _figsave import add_format_argument, save_figure  # noqa: E402
from psd_gig.fit_specs import FUNCTION_SPECS_BY_LABEL  # noqa: E402

FUNCTIONS = ["GIG_2p", "Lognormal", "Weibull"]
FUNCTION_LABELS = {"GIG_2p": "GIG", "Lognormal": "Lognormal", "Weibull": "Weibull"}
FUNCTION_COLORS = {"GIG_2p": "#C45755", "Lognormal": "#D9BC5E", "Weibull": "#5E769B"}

SET_COLORS = {
    "GIG_2p": "#C45755",
    "GIG_2p & Lognormal": "#E08A56",
    "GIG_2p & Lognormal & Weibull": "#8C4A46",
    "GIG_2p & Weibull": "#D9BC5E",
    "Lognormal": "#6BA292",
    "Lognormal & Weibull": "#8FB8C9",
    "Weibull": "#5E769B",
}
SET_LABELS = {
    "GIG_2p": "GIG",
    "GIG_2p & Lognormal": "GIG & Logn.",
    "GIG_2p & Lognormal & Weibull": "GIG & Logn. & Weib.",
    "GIG_2p & Weibull": "GIG & Weib.",
    "Lognormal": "Lognormal",
    "Lognormal & Weibull": "Logn. & Weib.",
    "Weibull": "Weibull",
}
CLASS_ORDER = ["coarse-skewed", "near-symmetric", "fine-skewed"]


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.4,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def style_axis(ax) -> None:
    ax.grid(True, axis="y", color="#E6E1D8", linewidth=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def violin_panel(ax, matrix: pd.DataFrame, metric: str, title: str) -> None:
    data = [matrix[label].dropna().to_numpy() for label in FUNCTIONS]
    positions = np.arange(len(FUNCTIONS))
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showextrema=False, showmedians=False)
    for body, label in zip(parts["bodies"], FUNCTIONS):
        body.set_facecolor(FUNCTION_COLORS[label])
        body.set_alpha(0.75)
        body.set_linewidth(0.3)
        body.set_edgecolor("white")

    low = float(np.nanpercentile(np.concatenate(data), 1.0))
    high = float(np.nanpercentile(np.concatenate(data), 99.0))
    span = high - low
    label_y = low - 0.19 * span
    ax.set_ylim(low - 0.30 * span, high + 0.06 * span)

    medians = [float(np.median(values)) for values in data]
    ax.scatter(positions, medians, s=9, color="black", zorder=4, linewidths=0)
    for position, median in zip(positions, medians):
        ax.text(position, label_y, f"{median:.2f}", ha="center", va="center", fontsize=6.4)

    ax.set_xticks(positions)
    ax.set_xticklabels([FUNCTION_LABELS[label] for label in FUNCTIONS])
    ax.set_ylabel(metric)
    ax.set_title(title, loc="left")
    style_axis(ax)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rhine", type=Path, default=ROOT / "outputs" / "rhine")
    parser.add_argument("--fits", type=Path,
                        default=ROOT / "outputs" / "fit_rhine" / "fitted_functions")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "Figure_S5")
    add_format_argument(parser)
    args = parser.parse_args()

    setup_style()
    samples = pd.read_csv(args.rhine / "rhine_sample_table.csv", dtype={"sample_ID": str})

    bic, rmse = {}, {}
    for label in FUNCTIONS:
        path = args.fits / FUNCTION_SPECS_BY_LABEL[label].output_name
        if not path.exists():
            raise FileNotFoundError(f"No Lower Rhine fit for {label}; run `make rhine` first")
        fit = pd.read_csv(path)
        fit["sample_ID"] = fit["sample_ID"].astype(str)
        fit = fit.drop_duplicates("sample_ID").set_index("sample_ID")
        bic[label] = fit["BIC"]
        rmse[label] = fit["RMSE"]

    figure = plt.figure(figsize=(7.1, 3.0))
    grid = GridSpec(1, 3, figure=figure, width_ratios=[1.0, 1.0, 1.35], wspace=0.38)
    ax_bic = figure.add_subplot(grid[0, 0])
    ax_rmse = figure.add_subplot(grid[0, 1])
    ax_bar = figure.add_subplot(grid[0, 2])

    violin_panel(ax_bic, pd.DataFrame(bic), "BIC", "a  Function performance, BIC")
    violin_panel(ax_rmse, pd.DataFrame(rmse), "RMSE", "b  Function performance, RMSE")

    counts = (samples.groupby(["skew_class", "best_function(s)"]).size()
              .unstack(fill_value=0).reindex(CLASS_ORDER).fillna(0))
    order = [name for name in SET_COLORS if name in counts.columns]
    counts = counts[order]
    bottom = np.zeros(len(counts))
    positions = np.arange(len(counts))
    for name in order:
        values = counts[name].to_numpy(float)
        ax_bar.bar(positions, values, bottom=bottom, width=0.66,
                   color=SET_COLORS[name], edgecolor="white", linewidth=0.7,
                   label=SET_LABELS[name])
        for position, value, base in zip(positions, values, bottom):
            if value >= 3:
                ax_bar.text(position, base + value / 2, f"{int(value)}", ha="center",
                            va="center", fontsize=6.2, color="white")
        bottom += values

    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([name.replace("-", "-\n") for name in counts.index])
    ax_bar.set_ylabel("Number of samples")
    ax_bar.set_xlabel("Folk-Ward skewness class")
    ax_bar.set_title("c  Best-fit set by skewness class", loc="left")
    # Headroom so the legend clears the tallest bar.
    ax_bar.set_ylim(0, bottom.max() * 1.62)
    ax_bar.legend(frameon=False, loc="upper right", handlelength=1.0, handleheight=0.9,
                  borderpad=0.15, labelspacing=0.32, fontsize=6.0)
    style_axis(ax_bar)

    figure.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.20)
    save_figure(figure, args.out, args.format, dpi=600, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
