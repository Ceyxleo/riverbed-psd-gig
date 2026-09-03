#!/usr/bin/env python3
"""Plot Google Scholar decade-count data for Supplementary Figure S1."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT / "data" / "reference" / "google_scholar_decade_counts.csv"
)
PLOT_DIR = ROOT / "figures"
MPLCONFIG_DIR = ROOT / "outputs" / ".mplconfig"
OUTPUT_STEM = "Figure_S1"

FUNCTION_ORDER = ["Lognormal", "Weibull", "GIG"]
FUNCTION_COLORS = {
    "Lognormal": "#4C78A8",
    "Weibull": "#F58518",
    "GIG": "#54A24B",
}


def load_decade_counts():
    import pandas as pd

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"function", "window_start", "window_end", "n_results"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["function"] = df["function"].astype(str).str.strip()
    df["window_start"] = df["window_start"].astype(int)
    df["window_end"] = df["window_end"].astype(int)
    df["n_results"] = (
        df["n_results"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": "0", "nan": "0"})
        .astype(float)
        .astype(int)
    )
    df["window"] = df["window_start"].astype(str) + "-" + df["window_end"].astype(str)
    df = df[df["window_start"] >= 1960].copy()

    unknown = sorted(set(df["function"]) - set(FUNCTION_ORDER))
    if unknown:
        raise ValueError(f"Unexpected function labels in input CSV: {unknown}")

    return df


def plot_decade_counts(df) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import MaxNLocator

    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    window_order = (
        df[["window_start", "window_end", "window"]]
        .drop_duplicates()
        .sort_values(["window_start", "window_end"])["window"]
        .tolist()
    )

    full_index = pd.MultiIndex.from_product(
        [window_order, FUNCTION_ORDER], names=["window", "function"]
    )
    plot_df = (
        df.groupby(["window", "function"], as_index=True)["n_results"]
        .sum()
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    pivot = plot_df.pivot(index="window", columns="function", values="n_results")
    pivot = pivot[FUNCTION_ORDER]

    x = np.arange(len(window_order))
    width = 0.24
    offsets = np.linspace(-width, width, len(FUNCTION_ORDER))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset, function in zip(offsets, FUNCTION_ORDER):
        bars = ax.bar(
            x + offset,
            pivot[function].to_numpy(),
            width=width,
            label=function,
            color=FUNCTION_COLORS[function],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{int(value)}" if value > 0 else "" for value in pivot[function]],
            padding=2,
            fontsize=8,
        )

    ax.set_ylabel("Number of Google Scholar results")
    ax.set_xlabel("Publication year")
    ax.set_xticks(x)
    ax.set_xticklabels(window_order, rotation=35, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    fig.tight_layout()
    png_path = PLOT_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    df = load_decade_counts()
    png_path = plot_decade_counts(df)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
