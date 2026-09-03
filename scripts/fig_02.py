from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "tables" / "analysis_frame.csv"
WBD_PATH = ROOT / "data" / "reference" / "wbd_hu2.gpkg"
OUT_PREFIX = ROOT / "figures" / "Figure_2"
CRS = "EPSG:5070"

FUNCTION_ORDER = ["G", "L", "W"]
FUNCTION_NAMES = {"G": "GIG", "L": "Lognormal", "W": "Weibull"}
FUNCTION_FULL_NAMES = {"G": "GIG_2p", "L": "Lognormal", "W": "Weibull"}
FUNCTION_COLORS = {"G": "#C45755", "L": "#D9BC5E", "W": "#5E769B"}
MAP_COLORS = {
    "G": FUNCTION_COLORS["G"],
    "L": FUNCTION_COLORS["L"],
    "W": FUNCTION_COLORS["W"],
}
REGION_COLORS = {"G": "#E9C7C5", "L": "#EFE4B8", "W": "#C9D4E5"}

SKEW_ORDER = ["SK1", "SK2", "SK3"]
SKEW_LABELS = {
    "SK1": "negative-skewed\n-1 to -0.1",
    "SK2": "near-symmetric\n-0.1 to 0.1",
    "SK3": "positive-skewed\n0.1 to 1",
}

D50_ORDER = ["sand_finer", "granule", "pebble", "coarse"]
D50_LABELS = {
    "sand_finer": "sand or finer\n<2 mm",
    "granule": "granule\n2-4 mm",
    "pebble": "pebble\n4-64 mm",
    "coarse": "coarse\n>=64 mm",
}

GRID_COLOR = "#E8E3DA"
TEXT_COLOR = "#222222"
AXIS_COLOR = "#666666"
MIN_SIGMA_BIN_COUNT = 50


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


def parse_functions(value: str) -> frozenset[str]:
    names = {part.strip() for part in str(value).split("&")}
    return frozenset(
        fn for fn, full_name in FUNCTION_FULL_NAMES.items() if full_name in names
    )


def skew_class(value: float) -> str | None:
    if -1 <= value < -0.1:
        return "SK1"
    if -0.1 <= value <= 0.1:
        return "SK2"
    if 0.1 < value <= 1:
        return "SK3"
    return None


def d50_class(value: float) -> str | None:
    if not np.isfinite(value):
        return None
    if value < 2:
        return "sand_finer"
    if value < 4:
        return "granule"
    if value < 64:
        return "pebble"
    return "coarse"


def huc2_from_huc_cd(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype("string").str.zfill(8).str[:2]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"site_no": "string", "huc_cd": "string"})
    for col in ["dec_long_va", "dec_lat_va", "skew_FW", "D50", "Sigma_g"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["functions"] = df["best_function(s)"].map(parse_functions)
    for fn in FUNCTION_ORDER:
        df[f"has_{fn}"] = df["functions"].map(lambda funcs, fn=fn: fn in funcs)
    df["n_functions"] = df["functions"].map(len)
    df["skew_group"] = df["skew_FW"].map(skew_class)
    df["d50_group"] = df["D50"].map(d50_class)
    df["huc2"] = huc2_from_huc_cd(df["huc_cd"])
    return df


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_COLOR)


def add_panel_label(ax: plt.Axes, label: str, x: float = 0.0, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="black",
    )


def pct(value: float) -> str:
    return f"{value:.0f}%"


def station_summary(data: pd.DataFrame) -> pd.DataFrame:
    valid = data.dropna(subset=["dec_long_va", "dec_lat_va"]).copy()
    station = (
        valid.groupby("site_no", dropna=False)
        .agg(
            dec_long_va=("dec_long_va", "first"),
            dec_lat_va=("dec_lat_va", "first"),
            huc2=("huc2", "first"),
            n_samples=("sample_ID", "count"),
            G_count=("has_G", "sum"),
            L_count=("has_L", "sum"),
            W_count=("has_W", "sum"),
        )
        .reset_index()
    )

    def category(row: pd.Series) -> str:
        counts = {"G": row["G_count"], "L": row["L_count"], "W": row["W_count"]}
        return max(FUNCTION_ORDER, key=lambda fn: counts[fn])

    station["map_category"] = station.apply(category, axis=1)
    return station


def regional_dominance(data: pd.DataFrame) -> dict[str, str]:
    rows = data[data["huc2"].notna()].copy()
    counts = (
        rows.groupby("huc2", dropna=False)[[f"has_{fn}" for fn in FUNCTION_ORDER]]
        .sum()
        .rename(columns={f"has_{fn}": fn for fn in FUNCTION_ORDER})
    )
    return {
        huc2: max(FUNCTION_ORDER, key=lambda fn: row[fn])
        for huc2, row in counts.iterrows()
    }


def load_wbd(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None

    wbd = gpd.read_file(path, layer="WBDHU2")
    wbd["huc2"] = wbd["huc2"].astype(str).str.zfill(2)
    return wbd.to_crs(CRS)


def plot_map_points(ax: plt.Axes, points: pd.DataFrame, size: float = 4.0) -> None:
    for category in ["L", "W", "G"]:
        subset = points[points["map_category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset["dec_long_va"],
            subset["dec_lat_va"],
            s=size,
            color=MAP_COLORS[category],
            edgecolors="none",
            alpha=0.88,
            zorder=3,
        )


def plot_map_fallback(ax: plt.Axes, data: pd.DataFrame) -> None:
    points = station_summary(data)
    conus = points[
        points["dec_long_va"].between(-126, -66)
        & points["dec_lat_va"].between(24, 50)
    ]
    plot_map_points(ax, conus, size=5.0)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_map_region(
    ax: plt.Axes,
    wbd,
    stations,
    huc2_values: list[str],
    region_dominance: dict[str, str],
    label_huc: bool = False,
    left_pad: float = 0.0,
) -> None:
    wbd_sub = wbd[wbd["huc2"].isin(huc2_values)].copy()
    stations_sub = stations[stations["huc2"].isin(huc2_values)]
    wbd_sub["dominant_fn"] = wbd_sub["huc2"].map(region_dominance)
    wbd_sub["face_color"] = wbd_sub["dominant_fn"].map(REGION_COLORS).fillna("#FFFFFF")

    wbd_sub.plot(
        ax=ax,
        color=wbd_sub["face_color"],
        edgecolor="#A7A7A7",
        linewidth=0.28,
        alpha=0.95,
        zorder=1,
    )
    for category in ["L", "W", "G"]:
        subset = stations_sub[stations_sub["map_category"] == category]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            markersize=2.2,
            color=MAP_COLORS[category],
            linewidth=0,
            alpha=0.84,
            zorder=3,
        )

    if len(wbd_sub) > 0:
        xmin, ymin, xmax, ymax = wbd_sub.total_bounds
        dx = xmax - xmin
        dy = ymax - ymin
        ax.set_xlim(xmin - dx * left_pad, xmax + dx * 0.02)
        ax.set_ylim(ymin - dy * 0.04, ymax + dy * 0.04)

    if label_huc:
        for _, row in wbd_sub.iterrows():
            point = row.geometry.representative_point()
            ax.annotate(
                row.huc2,
                xy=(point.x, point.y),
                ha="center",
                va="center",
                fontsize=4.6,
                color="#555555",
                zorder=2,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_map(ax: plt.Axes, data: pd.DataFrame, wbd) -> None:
    if wbd is None:
        plot_map_fallback(ax, data)
    else:
        import geopandas as gpd

        region_dom = regional_dominance(data)
        points = station_summary(data)
        stations = gpd.GeoDataFrame(
            points.copy(),
            geometry=gpd.points_from_xy(points.dec_long_va, points.dec_lat_va),
            crs="EPSG:4269",
        ).to_crs(CRS)
        boundary_huc = wbd[["huc2", "geometry"]].rename(columns={"huc2": "huc2_wbd"})
        joined = gpd.sjoin(stations, boundary_huc, how="left", predicate="intersects")
        joined["huc2"] = joined["huc2_wbd"].fillna(joined["huc2"]).astype(str).str.zfill(2)

        conus = [f"0{i}" for i in range(1, 10)] + [str(i) for i in range(10, 19)]
        plot_map_region(ax, wbd, joined, conus, region_dom, label_huc=True, left_pad=0.30)

        inset_specs = [
            ("19", [0.01, 0.60, 0.22, 0.36], "AK"),
            ("20", [-0.03, 0.22, 0.40, 0.44], "HI"),
            ("21", [0.20, 0.03, 0.17, 0.12], "PR"),
        ]
        for huc2, rect, label in inset_specs:
            iax = ax.inset_axes(rect)
            plot_map_region(iax, wbd, joined, [huc2], region_dom, label_huc=True)
            iax.text(0.03, 0.05, label, transform=iax.transAxes, fontsize=5.2)
            for spine in iax.spines.values():
                spine.set_visible(True)
                spine.set_color("#D0D0D0")
                spine.set_linewidth(0.55)

        try:
            from matplotlib_scalebar.scalebar import ScaleBar

            ax.add_artist(
                ScaleBar(
                    1,
                    units="m",
                    dimension="si-length",
                    location="lower right",
                    fixed_value=500,
                    fixed_units="km",
                    box_alpha=0,
                    scale_loc="bottom",
                    font_properties={"size": 5.2},
                )
            )
        except ImportError:
            pass

    add_panel_label(ax, "a")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MAP_COLORS["G"], markeredgewidth=0, markersize=4.4, label="GIG"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MAP_COLORS["L"], markeredgewidth=0, markersize=4.4, label="Lognormal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MAP_COLORS["W"], markeredgewidth=0, markersize=4.4, label="Weibull"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.52, -0.1),
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=5.9,
    )


def plot_overall(ax: plt.Axes, data: pd.DataFrame) -> None:
    y = np.arange(len(FUNCTION_ORDER))
    only = np.array(
        [
            ((data["n_functions"] == 1) & data[f"has_{fn}"]).mean() * 100
            for fn in FUNCTION_ORDER
        ]
    )
    tie = np.array(
        [
            ((data["n_functions"] > 1) & data[f"has_{fn}"]).mean() * 100
            for fn in FUNCTION_ORDER
        ]
    )
    totals = only + tie

    for i, fn in enumerate(FUNCTION_ORDER):
        ax.barh(
            y[i],
            only[i],
            height=0.48,
            color=FUNCTION_COLORS[fn],
            edgecolor="none",
        )
        ax.barh(
            y[i],
            tie[i],
            left=only[i],
            height=0.48,
            color=FUNCTION_COLORS[fn],
            edgecolor="none",
            alpha=0.36,
        )
        label_color = "white" if fn in {"G", "W"} else TEXT_COLOR
        ax.text(
            only[i] / 2,
            y[i],
            f"{only[i]:.0f}%",
            ha="center",
            va="center",
            fontsize=5.8,
            color=label_color,
        )
        ax.text(
            totals[i] + 1.8,
            y[i],
            pct(totals[i]),
            ha="left",
            va="center",
            fontsize=6.6,
            color=TEXT_COLOR,
            fontweight="bold" if fn == "G" else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([FUNCTION_NAMES[fn] for fn in FUNCTION_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Samples where function is best-fit (%)")
    add_panel_label(ax, "b")
    ax.text(
        0.07,
        0.96,
        f"n = {len(data):,} samples",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=AXIS_COLOR,
    )
    handles = [
        Patch(facecolor="#666666", edgecolor="none", label="solid: only best"),
        Patch(facecolor="#666666", edgecolor="none", alpha=0.36, label="pale: best tie"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.02),
        handlelength=1.0,
        labelspacing=0.25,
        fontsize=5.8,
    )
    style_axis(ax)


def grouped_percentages(data: pd.DataFrame, group_col: str, order: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    subset = data[data[group_col].isin(order)].copy()
    counts = subset.groupby(group_col, observed=False).size().reindex(order, fill_value=0)
    rows = []
    for group in order:
        group_data = subset[subset[group_col] == group]
        row = {}
        for fn in FUNCTION_ORDER:
            row[fn] = group_data[f"has_{fn}"].mean() * 100 if len(group_data) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=order), counts


def plot_grouped_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    group_col: str,
    order: list[str],
    labels: dict[str, str],
    panel_label: str,
    xlabel: str,
    show_ylabel: bool = False,
) -> None:
    table, counts = grouped_percentages(data, group_col, order)
    x = np.arange(len(order))
    width = 0.23

    for offset, fn in zip([-width, 0, width], FUNCTION_ORDER):
        values = table[fn].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=FUNCTION_COLORS[fn],
            edgecolor="white",
            linewidth=0.4,
            label=FUNCTION_NAMES[fn],
        )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            if value >= 18:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value - 4.2,
                    f"{value:.0f}",
                    ha="center",
                    va="top",
                    fontsize=5.7,
                    color="white" if fn != "L" else TEXT_COLOR,
                    fontweight="bold" if fn == "G" else "normal",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 2.0,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color=TEXT_COLOR,
                )

    for xi, group in zip(x, order):
        ax.text(
            xi,
            103,
            f"n={int(counts[group]):,}",
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=AXIS_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[group] for group in order])
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Best-fit frequency (%)")
    else:
        ax.set_yticklabels([])
    add_panel_label(ax, panel_label)
    style_axis(ax)


def sigma_bin_stats(data: pd.DataFrame, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    sigma = pd.to_numeric(data["Sigma_g"], errors="coerce")
    counts = np.zeros(len(edges) - 1, dtype=int)
    pcts = {fn: np.full(len(edges) - 1, np.nan) for fn in FUNCTION_ORDER}

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mask = (sigma >= lo) & (sigma <= hi)
        else:
            mask = (sigma >= lo) & (sigma < hi)
        subset = data[mask]
        counts[i] = len(subset)
        if counts[i] < MIN_SIGMA_BIN_COUNT:
            continue
        for fn in FUNCTION_ORDER:
            pcts[fn][i] = subset[f"has_{fn}"].mean() * 100

    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, counts, pcts


def plot_sigma(ax: plt.Axes, data: pd.DataFrame) -> None:
    edges = np.geomspace(1, 16, 18)
    centers, counts, pcts = sigma_bin_stats(data, edges)

    ax_count = ax.twinx()
    ax_count.bar(
        centers,
        counts,
        width=np.diff(edges) * 0.82,
        align="center",
        color="#CFCFCF",
        edgecolor="none",
        alpha=0.28,
        zorder=0,
    )
    ax_count.set_ylim(0, max(counts.max() * 3.0, 1))
    ax_count.set_yticks([])
    for spine in ax_count.spines.values():
        spine.set_visible(False)
    ax_count.grid(False)

    ax.set_zorder(ax_count.get_zorder() + 1)
    ax.patch.set_alpha(0)
    for fn in FUNCTION_ORDER:
        ax.plot(
            centers,
            pcts[fn],
            color=FUNCTION_COLORS[fn],
            linewidth=1.25,
            marker="o",
            markersize=2.5,
            label=FUNCTION_NAMES[fn],
        )

    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.set_xlabel(r"Geometric standard deviation, $\sigma_g$")
    add_panel_label(ax, "e")
    style_axis(ax)
    ax.text(
        0.98,
        0.06,
        f"gray bars: sample count\nlines omit bins with n<{MIN_SIGMA_BIN_COUNT}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=AXIS_COLOR,
    )


def make_figure(data_path: Path, wbd_path: Path | None, out_prefix: Path) -> None:
    setup_style()
    data = load_data(data_path)
    data = data[data["functions"].map(len) > 0].copy()
    skew_data = data[data["skew_group"].notna()].copy()
    wbd = load_wbd(wbd_path)

    fig = plt.figure(figsize=(7.8, 5.55))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1,1,1],
        height_ratios=[1.5, 1.0],
        hspace=0.25,
        wspace=0.15,
    )

    ax_map = fig.add_subplot(gs[0, 0:2])
    ax_overall = fig.add_subplot(gs[0, 2])
    ax_skew = fig.add_subplot(gs[1, 0])
    ax_d50 = fig.add_subplot(gs[1, 1])
    ax_sigma = fig.add_subplot(gs[1, 2])

    plot_map(ax_map, data, wbd)
    plot_overall(ax_overall, data)
    plot_grouped_bars(
        ax_skew,
        skew_data,
        "skew_group",
        SKEW_ORDER,
        SKEW_LABELS,
        "c",
        "Skewness",
        show_ylabel=True,
    )
    plot_grouped_bars(
        ax_d50,
        data,
        "d50_group",
        D50_ORDER,
        D50_LABELS,
        "d",
        r"Median grain size, $D_{50}$",
        show_ylabel=False,
    )
    plot_sigma(ax_sigma, data)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=600, facecolor="white")
    plt.close(fig)
    print(png_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redesigned Figure 2: best-fit frequency across sediment characteristics."
    )
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to DATA_final_all_optfuncs.csv")
    parser.add_argument("--wbd", default=str(WBD_PATH), help="Path to the WBDHU2 GeoPackage (data/reference/wbd_hu2.gpkg)")
    parser.add_argument("--no-wbd", action="store_true", help="Use lon/lat station scatter instead of WBD map")
    parser.add_argument("--out", default=str(OUT_PREFIX), help="Output prefix, without extension")
    args = parser.parse_args()

    make_figure(
        data_path=Path(args.data).expanduser(),
        wbd_path=None if args.no_wbd else Path(args.wbd).expanduser(),
        out_prefix=Path(args.out).expanduser(),
    )


if __name__ == "__main__":
    main()
