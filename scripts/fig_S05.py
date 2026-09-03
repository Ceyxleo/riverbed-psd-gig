#!/usr/bin/env python3
"""Supplementary Fig. S5 -- candidate-function performance on the Lower Rhine.

(a) BIC distribution across the 67 Lower Rhine samples for all 25 candidate
    functions, with the median printed under each violin.
(b) the same for RMSE.
(c) best-fit composition within each Folk-Ward skewness class, where the
    best-fit set holds every function within 2 BIC units of the sample minimum.

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
sys.path.insert(0, str(ROOT / "src"))

from psd_gig.fit_specs import FUNCTION_SPECS  # noqa: E402

TWO_PARAMETER_COLOR = "#5E769B"
THREE_PARAMETER_COLOR = "#C45755"
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
        "font.size": 6.4, "axes.labelsize": 6.4, "axes.titlesize": 6.8,
        "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.6,
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


def violin_panel(ax, matrix: pd.DataFrame, labels: list[str], n_params: dict[str, int],
                 metric: str, title: str, label_y: float) -> None:
    data = [matrix[label].dropna().to_numpy() for label in labels]
    positions = np.arange(len(labels))
    parts = ax.violinplot(data, positions=positions, widths=0.82,
                          showextrema=False, showmedians=False)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(THREE_PARAMETER_COLOR if n_params[label] == 3
                           else TWO_PARAMETER_COLOR)
        body.set_alpha(0.72)
        body.set_linewidth(0.2)
        body.set_edgecolor("white")
    medians = [np.median(values) for values in data]
    ax.scatter(positions, medians, s=5.5, color="black", zorder=4, linewidths=0)
    for position, median in zip(positions, medians):
        ax.text(position, label_y, f"{median:.2f}", ha="center", va="bottom",
                fontsize=5.0, rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
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
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both")
    args = parser.parse_args()

    setup_style()
    samples = pd.read_csv(args.rhine / "rhine_sample_table.csv", dtype={"sample_ID": str})

    bic, rmse = {}, {}
    n_params = {}
    for spec in FUNCTION_SPECS:
        matches = sorted(args.fits.glob(f"F{spec.number:02d}_*.csv"))
        if not matches:
            continue
        fit = pd.read_csv(matches[0])
        fit["sample_ID"] = fit["sample_ID"].astype(str)
        fit = fit.drop_duplicates("sample_ID").set_index("sample_ID")
        bic[spec.label] = fit["BIC"]
        rmse[spec.label] = fit["RMSE"]
        n_params[spec.label] = spec.n_params
    labels = list(bic)
    bic_matrix = pd.DataFrame(bic)
    rmse_matrix = pd.DataFrame(rmse)

    figure = plt.figure(figsize=(7.1, 4.6))
    grid = GridSpec(2, 3, figure=figure, width_ratios=[1.0, 1.0, 0.72],
                    hspace=0.62, wspace=0.32)
    ax_bic = figure.add_subplot(grid[0, 0:2])
    ax_rmse = figure.add_subplot(grid[1, 0:2])
    ax_bar = figure.add_subplot(grid[:, 2])

    bic_low = float(np.nanpercentile(bic_matrix.to_numpy(), 0.5))
    bic_high = float(np.nanpercentile(bic_matrix.to_numpy(), 99.0))
    violin_panel(ax_bic, bic_matrix, labels, n_params, "BIC",
                 "a  Function performance, BIC", bic_low - 0.30 * (bic_high - bic_low))
    ax_bic.set_ylim(bic_low - 0.32 * (bic_high - bic_low), bic_high)
    ax_bic.set_xticklabels([])

    rmse_high = float(np.nanpercentile(rmse_matrix.to_numpy(), 99.0))
    violin_panel(ax_rmse, rmse_matrix, labels, n_params, "RMSE",
                 "b  Function performance, RMSE", -0.30 * rmse_high)
    ax_rmse.set_ylim(-0.32 * rmse_high, rmse_high)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TWO_PARAMETER_COLOR, alpha=0.72),
               plt.Rectangle((0, 0), 1, 1, color=THREE_PARAMETER_COLOR, alpha=0.72)]
    ax_bic.legend(handles, ["two-parameter", "three-parameter"], loc="upper right",
                  frameon=False, ncol=2, handlelength=1.1)

    counts = (samples.groupby(["skew_class", "best_function(s)"]).size()
              .unstack(fill_value=0).reindex(CLASS_ORDER).fillna(0))
    order = [name for name in SET_COLORS if name in counts.columns]
    counts = counts[order]
    bottom = np.zeros(len(counts))
    positions = np.arange(len(counts))
    for name in order:
        values = counts[name].to_numpy(float)
        ax_bar.bar(positions, values, bottom=bottom, width=0.74,
                   color=SET_COLORS[name], edgecolor="white", linewidth=0.7,
                   label=SET_LABELS[name])
        for position, value, base in zip(positions, values, bottom):
            if value >= 3:
                ax_bar.text(position, base + value / 2, f"{int(value)}", ha="center",
                            va="center", fontsize=5.4, color="white")
        bottom += values
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([name.replace("-", "-\n") for name in counts.index])
    ax_bar.set_ylabel("Number of samples")
    ax_bar.set_xlabel("Folk-Ward skewness class")
    ax_bar.set_title("c  Best-fit set by\n    skewness class", loc="left")
    ax_bar.legend(frameon=False, loc="upper right", handlelength=1.0, borderpad=0.2)
    style_axis(ax_bar)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if args.format in {"png", "both"}:
        figure.savefig(args.out.with_suffix(".png"), dpi=600, bbox_inches="tight")
        written.append(args.out.with_suffix(".png"))
    if args.format in {"pdf", "both"}:
        figure.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
        written.append(args.out.with_suffix(".pdf"))
    plt.close(figure)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
