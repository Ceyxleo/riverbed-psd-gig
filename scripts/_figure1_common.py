from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import geninvgauss


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "tables" / "analysis_frame.csv"
SIZE_PATH = ROOT / "data" / "reference" / "sieve_sizes_all43.csv"
GIG_PATH = ROOT / "data" / "fitted_functions" / "fits_F14_GIG_2p.csv"
LOGNORMAL_PATH = ROOT / "data" / "fitted_functions" / "fits_F10_Lognormal.csv"
WEIBULL_PATH = ROOT / "data" / "fitted_functions" / "fits_F15_Weibull.csv"
OUT_PREFIX = ROOT / "figures" / "Figure_1_skewness_quantile_half"

FUNCTION_ORDER = ["GIG_2p", "Lognormal", "Weibull"]
FUNCTION_LABELS = {
    "GIG_2p": "GIG",
    "Lognormal": "Lognormal",
    "Weibull": "Weibull",
}
FUNCTION_COLORS = {
    "GIG_2p": "#D55E00",
    "Lognormal": "#009E73",
    "Weibull": "#0072B2",
}
FUNCTION_STYLES = {
    "GIG_2p": "-",
    "Lognormal": "--",
    "Weibull": ":",
}

GROUP_ORDER = ["negative", "symmetric", "positive"]
GROUP_LABELS = {
    "negative": "coarse-skewed tail\nskew <= P10",
    "symmetric": "near-symmetric\n-0.1 <= skew <= 0.1",
    "positive": "fine-skewed tail\nskew >= P90",
}

GRID_COLOR = "#E7E7E7"
AXIS_COLOR = "#666666"
TEXT_COLOR = "#222222"
OBS_LINE_COLOR = "#222222"
OBS_BAND_COLOR = "#BDBDBD"
LOG2_TICKS = np.array([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])


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
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.45)
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
        color="black",
    )


def load_size_lookup(path: Path) -> pd.Series:
    sizes = pd.read_csv(path).iloc[0]
    sizes = pd.to_numeric(sizes, errors="coerce")
    return sizes.dropna().sort_values()


def load_data(data_path: Path, size_lookup: pd.Series) -> pd.DataFrame:
    dtypes = {"site_no": "string"}
    data = pd.read_csv(data_path, dtype=dtypes)
    for col in ["sample_ID", "skew_FW", "D50"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in size_lookup.index:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["site_no", "sample_ID", "skew_FW"]).copy()


def make_groups(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    p10 = data["skew_FW"].quantile(0.10)
    p90 = data["skew_FW"].quantile(0.90)
    groups = {
        "negative": data[data["skew_FW"] <= p10].copy(),
        "symmetric": data[data["skew_FW"].between(-0.1, 0.1, inclusive="both")].copy(),
        "positive": data[data["skew_FW"] >= p90].copy(),
    }
    return groups, {"p10": float(p10), "p90": float(p90)}


def observed_cdf_on_grid(
    row: pd.Series,
    size_lookup: pd.Series,
    log2_grid: np.ndarray,
    center_by_d50: bool,
) -> np.ndarray:
    values = pd.to_numeric(row[size_lookup.index], errors="coerce").to_numpy(dtype=float)
    sizes = size_lookup.to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(sizes)
    if valid.sum() < 2:
        return np.full_like(log2_grid, np.nan, dtype=float)

    if center_by_d50:
        d50 = row["D50"]
        if not np.isfinite(d50) or d50 <= 0:
            return np.full_like(log2_grid, np.nan, dtype=float)
        x = np.log2(sizes[valid] / d50)
    else:
        x = np.log2(sizes[valid])
    y = np.clip(values[valid], 0.0, 100.0)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) < 2:
        return np.full_like(log2_grid, np.nan, dtype=float)

    unique_y = np.array([np.nanmean(y[inverse == i]) for i in range(len(unique_x))])
    unique_y = np.maximum.accumulate(unique_y)
    unique_y = np.clip(unique_y, 0.0, 100.0)
    return np.interp(log2_grid, unique_x, unique_y, left=np.nan, right=np.nan)


def cdf_to_density(cdf: np.ndarray, log2_grid: np.ndarray) -> np.ndarray:
    density = np.full_like(cdf, np.nan, dtype=float)
    finite = np.isfinite(cdf)
    if finite.sum() < 3:
        return density
    idx = np.where(finite)[0]
    segment = np.gradient(cdf[idx], log2_grid[idx])
    density[idx] = np.clip(segment, 0.0, np.inf)
    return density


def observed_density_matrix(
    data: pd.DataFrame,
    size_lookup: pd.Series,
    log2_grid: np.ndarray,
    center_by_d50: bool,
) -> np.ndarray:
    curves = []
    for _, row in data.iterrows():
        cdf = observed_cdf_on_grid(row, size_lookup, log2_grid, center_by_d50)
        curves.append(cdf_to_density(cdf, log2_grid))
    return np.vstack(curves)


def read_fit(path: Path) -> pd.DataFrame:
    fit = pd.read_csv(path, dtype={"site_no": "string"})
    fit["sample_ID"] = pd.to_numeric(fit["sample_ID"], errors="coerce")
    return fit


def load_fits() -> dict[str, pd.DataFrame]:
    return {
        "GIG_2p": read_fit(GIG_PATH),
        "Lognormal": read_fit(LOGNORMAL_PATH),
        "Weibull": read_fit(WEIBULL_PATH),
    }


def merge_group_fit(group: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    keys = ["site_no", "sample_ID"]
    return group[keys + ["D50"]].merge(fit, on=keys, how="inner")


def eval_function(function_name: str, x: np.ndarray, params: pd.Series) -> np.ndarray:
    a = params["fitted_A"]
    b = params["fitted_B"]
    if function_name == "GIG_2p":
        return geninvgauss.cdf(x, a, b) * 100.0
    if function_name == "Lognormal":
        return 0.5 * (1.0 + special.erf((np.log(x) - b) / (a * np.sqrt(2.0)))) * 100.0
    if function_name == "Weibull":
        return (1.0 - np.exp(-((x / a) ** b))) * 100.0
    raise ValueError(f"Unknown function: {function_name}")


def eval_density(function_name: str, x: np.ndarray, params: pd.Series) -> np.ndarray:
    a = params["fitted_A"]
    b = params["fitted_B"]
    jacobian = x * np.log(2.0)
    if function_name == "GIG_2p":
        return geninvgauss.pdf(x, a, b) * jacobian * 100.0
    if function_name == "Lognormal":
        if a <= 0:
            return np.full_like(x, np.nan, dtype=float)
        sigma = a
        mu = b
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pdf_x = np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2) / (
                x * sigma * np.sqrt(2.0 * np.pi)
            )
        return pdf_x * jacobian * 100.0
    if function_name == "Weibull":
        if a <= 0 or b <= 0:
            return np.full_like(x, np.nan, dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pdf_x = (b / a) * (x / a) ** (b - 1.0) * np.exp(-((x / a) ** b))
        return pdf_x * jacobian * 100.0
    raise ValueError(f"Unknown function: {function_name}")


def fitted_density_matrix(
    group: pd.DataFrame,
    fit: pd.DataFrame,
    function_name: str,
    x_grid: np.ndarray,
    center_by_d50: bool,
) -> np.ndarray:
    merged = merge_group_fit(group, fit)
    curves = []
    for _, row in merged.iterrows():
        if center_by_d50:
            d50 = row["D50"]
            if not np.isfinite(d50) or d50 <= 0:
                continue
            eval_x = x_grid * d50
        else:
            eval_x = x_grid
        density = eval_density(function_name, eval_x, row)
        density = np.asarray(density, dtype=float)
        density[~np.isfinite(density)] = np.nan
        curves.append(np.clip(density, 0.0, np.inf))
    if not curves:
        return np.empty((0, len(x_grid)))
    return np.vstack(curves)


def summarize_matrix(matrix: np.ndarray, min_count: int) -> pd.DataFrame:
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    summary = pd.DataFrame(
        {
            "mean": np.nanmean(matrix, axis=0),
            "q25": np.nanpercentile(matrix, 25, axis=0),
            "q75": np.nanpercentile(matrix, 75, axis=0),
            "n": counts,
        }
    )
    mask = counts >= min_count
    for col in ["mean", "q25", "q75"]:
        summary.loc[~mask, col] = np.nan
    return summary


def write_summary(
    path: Path,
    groups: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
) -> None:
    rows = []
    for group_name in GROUP_ORDER:
        group = groups[group_name]
        rows.append(
            {
                "group": group_name,
                "label": GROUP_LABELS[group_name].replace("\n", " "),
                "n_samples": len(group),
                "skew_min": group["skew_FW"].min(),
                "skew_median": group["skew_FW"].median(),
                "skew_max": group["skew_FW"].max(),
                "D50_median_mm": group["D50"].median(),
                "skew_p10_threshold": thresholds["p10"],
                "skew_p90_threshold": thresholds["p90"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_figure(
    groups: dict[str, pd.DataFrame],
    fits: dict[str, pd.DataFrame],
    size_lookup: pd.Series,
    out_prefix: Path,
    n_grid: int,
    min_fraction: float,
    center_by_d50: bool,
) -> None:
    setup_style()
    if center_by_d50:
        log2_grid = np.linspace(-6.0, 6.0, n_grid)
        x_grid = 2.0**log2_grid
        xticks = np.array([1 / 64, 1 / 16, 1 / 4, 1, 4, 16, 64], dtype=float)
        xticklabels = ["1/64", "1/16", "1/4", "1", "4", "16", "64"]
        x_label = r"Relative particle size ($D/D_{50}$, log2 scale)"
        y_label = r"Probability density (% per log2 $D/D_{50}$)"
    else:
        log2_grid = np.linspace(np.log2(0.0625), np.log2(128.0), n_grid)
        x_grid = 2.0**log2_grid
        xticks = LOG2_TICKS
        xticklabels = [
            "0.06",
            "0.13",
            "0.25",
            "0.5",
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",
        ]
        x_label = "Particle size (mm, log2 scale)"
        y_label = "Probability density (% per log2 mm)"

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.2), sharex=True, sharey=True)
    panel_letters = ["a", "b", "c"]

    for ax, group_name, panel_letter in zip(axes, GROUP_ORDER, panel_letters):
        group = groups[group_name]
        min_count = max(20, int(np.ceil(len(group) * min_fraction)))

        obs_matrix = observed_density_matrix(
            group,
            size_lookup,
            log2_grid,
            center_by_d50=center_by_d50,
        )
        obs_summary = summarize_matrix(obs_matrix, min_count=min_count)

        ax.fill_between(
            x_grid,
            obs_summary["q25"],
            obs_summary["q75"],
            color=OBS_BAND_COLOR,
            alpha=0.32,
            linewidth=0,
            label="observed IQR",
        )
        ax.plot(
            x_grid,
            obs_summary["mean"],
            color=OBS_LINE_COLOR,
            linewidth=1.25,
            label="observed mean",
        )

        for function_name in FUNCTION_ORDER:
            matrix = fitted_density_matrix(
                group,
                fits[function_name],
                function_name,
                x_grid,
                center_by_d50=center_by_d50,
            )
            fit_summary = summarize_matrix(matrix, min_count=min_count)
            ax.plot(
                x_grid,
                fit_summary["mean"],
                color=FUNCTION_COLORS[function_name],
                linestyle=FUNCTION_STYLES[function_name],
                linewidth=1.15,
                label=FUNCTION_LABELS[function_name],
            )

        style_axis(ax)
        add_panel_label(ax, panel_letter)
        ax.set_title(
            f"{GROUP_LABELS[group_name]}\n"
            f"n={len(group):,}; median skew={group['skew_FW'].median():.2f}",
            color=TEXT_COLOR,
            pad=8,
        )
        ax.set_xscale("log", base=2)
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel(y_label)
    for ax in axes:
        ax.set_xlabel(x_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.065),
        handlelength=2.4,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.075, right=0.995, top=0.78, bottom=0.32, wspace=0.19)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=600)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Figure 1 prototype: observed PSD density envelopes and average fitted "
            "curves for extreme and near-symmetric skewness groups."
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--sizes", type=Path, default=SIZE_PATH)
    parser.add_argument("--out-prefix", type=Path, default=OUT_PREFIX)
    parser.add_argument("--n-grid", type=int, default=600)
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.08,
        help="Minimum fraction of group samples required at a grain-size grid point.",
    )
    parser.add_argument(
        "--center-by-d50",
        action="store_true",
        help="Center every observed and fitted curve by the sample D50 before averaging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size_lookup = load_size_lookup(args.sizes)
    data = load_data(args.data, size_lookup)
    groups, thresholds = make_groups(data)
    fits = load_fits()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_figure(
        groups=groups,
        fits=fits,
        size_lookup=size_lookup,
        out_prefix=args.out_prefix,
        n_grid=args.n_grid,
        min_fraction=args.min_fraction,
        center_by_d50=args.center_by_d50,
    )
    write_summary(
        args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv"),
        groups,
        thresholds,
    )


if __name__ == "__main__":
    main()
