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
from matplotlib.patches import Patch
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsave import add_format_argument, save_figure  # noqa: E402

DEFAULT_METRICS = ROOT / "outputs" / "water" / "sample_water_metrics.csv"
DEFAULT_BIC_ONE = ROOT / "data" / "usgs_sample_statistics.csv"
DEFAULT_OUT = ROOT / "figures" / "Figure_3"

FIT_FILES = {
    "GIG_2p": ROOT / "data" / "fitted_functions" / "fits_F14_GIG_2p.csv",
    "Lognormal": ROOT / "data" / "fitted_functions" / "fits_F10_Lognormal.csv",
    "Weibull": ROOT / "data" / "fitted_functions" / "fits_F15_Weibull.csv",
}

FUNCTION_ORDER = ["GIG_2p", "Lognormal", "Weibull"]
FUNCTION_LABELS = {"GIG_2p": "GIG", "Lognormal": "Lognormal", "Weibull": "Weibull"}
FUNCTION_COLORS = {
    "GIG_2p": "#C45755",
    "Lognormal": "#D9BC5E",
    "Weibull": "#5E769B",
}

PARAMETERS = {
    "GIG_2p": [
        ("eta", r"$\eta$", "fitted_A"),
        ("beta", r"$\beta$", "fitted_B"),
    ],
    "Lognormal": [
        ("mu", r"$\mu$", "fitted_B"),
        ("sigma", r"$\sigma$", "fitted_A"),
    ],
    "Weibull": [
        ("k", r"$k$", "fitted_B"),
        ("lambda", r"$\lambda$", "fitted_A"),
    ],
}

TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"
GRID_COLOR = "#E6E1D8"
THRESHOLD_COLOR = "#5F5B55"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def function_is_best(best_functions: pd.Series, function_name: str) -> pd.Series:
    return best_functions.astype(str).str.contains(function_name, regex=False, na=False)


def load_hydro(metrics_path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        metrics_path,
        usecols=["sample_ID", "site_no", "best_function(s)", "Re_reference", "Re_shear_reference"],
        dtype={"site_no": str},
    )
    data = data.rename(columns={"Re_reference": "Re", "Re_shear_reference": "Re_star"})
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Re", "Re_star"])
    data = data[(data["Re"] > 0) & (data["Re_star"] > 0)].copy()
    data["log10_Re"] = np.log10(data["Re"])
    return data


def load_fit(function_name: str) -> pd.DataFrame:
    fit = pd.read_csv(FIT_FILES[function_name], usecols=["sample_ID", "fitted_A", "fitted_B"])
    return fit.replace([np.inf, -np.inf], np.nan)


def parameter_table(hydro: pd.DataFrame, function_name: str, best_only: bool = True) -> pd.DataFrame:
    fit = load_fit(function_name)
    data = hydro.merge(fit, on="sample_ID", how="inner")
    if best_only:
        data = data[function_is_best(data["best_function(s)"], function_name)].copy()

    for parameter, _, source_col in PARAMETERS[function_name]:
        data[parameter] = pd.to_numeric(data[source_col], errors="coerce")

    required = [parameter for parameter, _, _ in PARAMETERS[function_name]]
    data = data.dropna(subset=required).copy()
    return data


def build_correlations(hydro: pd.DataFrame, best_only: bool = True) -> pd.DataFrame:
    rows = []
    for function_name in FUNCTION_ORDER:
        data = parameter_table(hydro, function_name, best_only=best_only)
        for parameter, parameter_label, _ in PARAMETERS[function_name]:
            clean = data.dropna(subset=[parameter, "Re", "Re_star"])
            for variable, variable_label in [("Re_star", "Re*"), ("Re", "Re")]:
                rho, p_value = stats.spearmanr(clean[parameter], clean[variable])
                rows.append(
                    {
                        "function": function_name,
                        "function_label": FUNCTION_LABELS[function_name],
                        "parameter": parameter,
                        "parameter_label": parameter_label,
                        "hydraulic_variable": variable,
                        "hydraulic_variable_label": variable_label,
                        "n": int(len(clean)),
                        "n_sites": int(clean["site_no"].nunique()),
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                    }
                )
    return pd.DataFrame(rows)


def classify_hydraulic_regime(re_star: pd.Series) -> pd.Series:
    regimes = pd.Series(index=re_star.index, dtype="object")
    regimes.loc[re_star < 5] = "smooth"
    regimes.loc[(re_star >= 5) & (re_star < 70)] = "transitional"
    regimes.loc[re_star >= 70] = "rough"
    return regimes


def build_regime_winners(hydro: pd.DataFrame, bic_one_path: Path) -> pd.DataFrame:
    winners = pd.read_csv(bic_one_path, usecols=["sample_ID", "best_of_three_by_BIC"])
    winners = winners.rename(columns={"best_of_three_by_BIC": "Function"})
    data = hydro[["sample_ID", "site_no", "Re_star"]].merge(winners, on="sample_ID", how="inner")
    data = data[data["Function"].isin(FUNCTION_ORDER)].copy()
    data["regime"] = classify_hydraulic_regime(data["Re_star"])
    data = data.dropna(subset=["regime"]).copy()

    regime_order = ["smooth", "transitional", "rough"]
    regime_labels = {
        "smooth": r"Smooth" + "\n" + r"Re$^{*}$ < 5",
        "transitional": r"Transitional" + "\n" + r"5 $\leq$ Re$^{*}$ < 70",
        "rough": r"Rough" + "\n" + r"Re$^{*}$ $\geq$ 70",
    }

    rows = []
    for regime in regime_order:
        subset = data[data["regime"] == regime]
        total = len(subset)
        for function_name in FUNCTION_ORDER:
            count = int((subset["Function"] == function_name).sum())
            rows.append(
                {
                    "regime": regime,
                    "regime_label": regime_labels[regime],
                    "function": function_name,
                    "function_label": FUNCTION_LABELS[function_name],
                    "count": count,
                    "share_percent": count / total * 100 if total else np.nan,
                    "n": int(total),
                    "n_sites": int(subset["site_no"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_COLOR)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def add_restar_thresholds(ax: plt.Axes, labels: bool = False) -> None:
    for threshold in [5, 70]:
        ax.axvline(threshold, color=THRESHOLD_COLOR, linewidth=0.8, linestyle=(0, (2, 1.6)), zorder=3)
    if labels:
        trans = ax.get_xaxis_transform()
        ax.text(2.6, 1.018, "smooth", transform=trans, ha="center", va="bottom", fontsize=5.4, clip_on=False)
        ax.text(18.5, 1.018, "transitional", transform=trans, ha="center", va="bottom", fontsize=5.4, clip_on=False)
        ax.text(2400, 1.018, "rough", transform=trans, ha="center", va="bottom", fontsize=5.4, clip_on=False)


def draw_gig_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    parameter: str,
    ylabel: str,
    panel: str,
    marker: str,
    cmap: mpl.colors.Colormap,
    norm: Normalize,
    correlations: pd.DataFrame,
    show_regime_labels: bool = False,
) -> mpl.collections.PathCollection:
    points = ax.scatter(
        data["Re_star"],
        data[parameter],
        c=data["log10_Re"],
        cmap=cmap,
        norm=norm,
        s=4.9,
        marker=marker,
        linewidths=0,
        alpha=0.50,
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Shear Reynolds number, Re$^{*}$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.8, 1.5e5)
    style_axis(ax)
    add_restar_thresholds(ax, labels=show_regime_labels)
    add_panel_label(ax, panel)

    rho = correlations[
        (correlations["function"] == "GIG_2p")
        & (correlations["parameter"] == parameter)
        & (correlations["hydraulic_variable"] == "Re_star")
    ]["spearman_rho"].iloc[0]
    ax.text(
        0.035,
        0.88,
        rf"$\rho_s$ = {rho:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
    )
    return points


def draw_regime_winner_panel(ax: plt.Axes, regime_winners: pd.DataFrame) -> None:
    regime_order = ["smooth", "transitional", "rough"]
    y_positions = np.arange(len(regime_order))[::-1]
    label_lookup = (
        regime_winners[["regime", "regime_label"]].drop_duplicates().set_index("regime")["regime_label"].to_dict()
    )

    for y, regime in zip(y_positions, regime_order):
        left = 0.0
        subset = regime_winners[regime_winners["regime"] == regime].set_index("function")
        for function_name in FUNCTION_ORDER:
            share = float(subset.loc[function_name, "share_percent"])
            ax.barh(
                y,
                share,
                left=left,
                height=0.58,
                color=FUNCTION_COLORS[function_name],
                edgecolor="white",
                linewidth=0.8,
            )
            if share >= 8:
                ax.text(
                    left + share / 2,
                    y,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    fontweight="bold" if function_name == "GIG_2p" else "normal",
                    color=TEXT_COLOR,
                )
            left += share

        n = int(subset["n"].iloc[0])
        ax.text(101.3, y, f"n={n:,}", ha="left", va="center", fontsize=5.7, color=TEXT_COLOR)

    ax.set_xlim(0, 112)
    ax.set_ylim(-0.55, len(regime_order) - 0.45)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label_lookup[regime] for regime in regime_order])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Lowest-BIC samples (%)")
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.45)
    ax.grid(False, axis="y")
    ax.tick_params(colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, "c")

    handles = [
        Patch(facecolor=FUNCTION_COLORS[function_name], edgecolor="none", label=FUNCTION_LABELS[function_name])
        for function_name in FUNCTION_ORDER
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.24),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.9,
    )


def make_figure(
    metrics_path: Path,
    bic_one_path: Path,
    out_prefix: Path,
    best_only: bool = True,
    fmt: str = "png",
) -> None:
    setup_style()
    hydro = load_hydro(metrics_path)
    correlations = build_correlations(hydro, best_only=best_only)
    regime_winners = build_regime_winners(hydro, bic_one_path)
    gig = parameter_table(hydro, "GIG_2p", best_only=best_only)
    gig = gig.dropna(subset=["eta", "beta", "Re", "Re_star", "log10_Re"]).copy()
    gig = gig[(gig["beta"] > 0) & (gig["Re"] > 0) & (gig["Re_star"] > 0)].copy()

    color_values = gig["log10_Re"]
    norm = Normalize(vmin=np.floor(color_values.quantile(0.01)), vmax=np.ceil(color_values.quantile(0.99)))
    cmap = LinearSegmentedColormap.from_list(
        "re_purple",
        ["#F7F3FA", "#D9C7E7", "#A880C4", "#6D4A9B", "#3B226A"],
    )

    fig = plt.figure(figsize=(6.9, 4.45))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.25, 0.55],
        hspace=0.50,
        wspace=0.30,
    )
    ax_eta = fig.add_subplot(gs[0, 0])
    ax_beta = fig.add_subplot(gs[0, 1])
    cax = ax_beta.inset_axes([1.035, 0.0, 0.04, 1.0])
    ax_regime = fig.add_subplot(gs[1, :])

    points = draw_gig_scatter(
        ax_eta,
        gig,
        "eta",
        r"GIG parameter $\eta$",
        "a",
        "o",
        cmap,
        norm,
        correlations,
        show_regime_labels=True,
    )
    draw_gig_scatter(
        ax_beta,
        gig,
        "beta",
        r"GIG parameter $\beta$",
        "b",
        "^",
        cmap,
        norm,
        correlations,
        show_regime_labels=False,
    )

    ax_eta.set_yscale("symlog", linthresh=1)
    ax_eta.set_ylim(gig["eta"].quantile(0.001), gig["eta"].quantile(0.999))
    ax_beta.set_yscale("log")
    ax_beta.set_ylim(gig["beta"].quantile(0.001), gig["beta"].quantile(0.999))

    cbar = fig.colorbar(points, cax=cax)
    cbar.set_label(r"$\log_{10}$(channel Reynolds number, Re)", rotation=270, labelpad=12)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=5.8, colors=AXIS_COLOR)

    draw_regime_winner_panel(ax_regime, regime_winners)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_prefix, fmt, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 3 hydraulic-regime redesign.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--bic-one", type=Path, default=DEFAULT_BIC_ONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--all-fits",
        action="store_true",
        help="Use all fitted samples for each function instead of samples where the function is selected as best fit.",
    )
    add_format_argument(parser)
    args = parser.parse_args()
    make_figure(
        args.metrics.expanduser(),
        args.bic_one.expanduser(),
        args.out.expanduser(),
        best_only=not args.all_fits,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
