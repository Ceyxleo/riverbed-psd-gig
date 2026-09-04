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


ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsave import add_format_argument, save_figure  # noqa: E402

DATA_PATH = ROOT / "outputs" / "tables" / "analysis_frame.csv"
WBD_PATH = ROOT / "data" / "reference" / "wbd_hu2.gpkg"
OUT_PREFIX = ROOT / "figures" / "Figure_S2"
CRS = "EPSG:5070"

GRAIN_ORDER = ["sand", "granule", "pebble", "cobble"]
GRAIN_LABELS = {
    "sand": "sand\n<2 mm",
    "granule": "granule\n2-4 mm",
    "pebble": "pebble\n4-64 mm",
    "cobble": "cobble\n>=64 mm",
}
GRAIN_MARKERS = {
    "sand": "o",
    "granule": "^",
    "pebble": "s",
    "cobble": "D",
}

GRID_COLOR = "#E6E6E6"
TEXT_COLOR = "#222222"
AXIS_COLOR = "#666666"
BOUNDARY_COLOR = "#B8B8B8"
WBD_FACE_COLOR = "#F8F8F6"
SAMPLE_TICKS = [1, 2, 5, 10, 25, 50, 100, 250, 750]
SAMPLE_LINE_COLOR = "#2F6F83"
D50_LINE_COLOR = "#9A565C"
SAMPLE_CMAP_COLORS = ["#F7FAFC", "#D9ECF1", "#A8D4DA", "#6EAEB8", SAMPLE_LINE_COLOR]
SAMPLE_COUNT_EDGES = np.array([1, 2, 5, 10, 25, 50, 100, 250, 751], dtype=float)
D50_EDGES = np.array(
    [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128], dtype=float
)


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


def sample_count_cmap() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "sample_count_light", SAMPLE_CMAP_COLORS
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS_COLOR, labelcolor=AXIS_COLOR)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_COLOR)


def add_panel_label(ax: plt.Axes, label: str, x: float = 0.0, y: float = 1.03) -> None:
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


def add_figure_panel_labels(
    fig: plt.Figure,
    ax_samples: plt.Axes,
    ax_d50: plt.Axes,
) -> None:
    label_style = {
        "ha": "left",
        "va": "top",
        "fontsize": 9,
        "fontweight": "bold",
        "color": "black",
    }
    x_left = fig.subplotpars.left
    hist_top = max(ax_samples.get_position().y1, ax_d50.get_position().y1)
    fig.text(0.065, 0.955, "a", ha="left", va="top", fontsize=9, fontweight="bold")
    fig.text(0.065, 0.385, "b", ha="left", va="top", fontsize=9, fontweight="bold")
    fig.text(0.535, 0.385, "c", ha="left", va="top", fontsize=9, fontweight="bold")


def huc2_from_huc_cd(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    huc2 = cleaned.str.zfill(8).str[:2]
    return huc2.where(cleaned.notna() & (cleaned != ""))


def grain_class(d50: float) -> str | None:
    if not np.isfinite(d50):
        return None
    if d50 < 2:
        return "sand"
    if d50 < 4:
        return "granule"
    if d50 < 64:
        return "pebble"
    return "cobble"


def load_samples(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        dtype={
            "MonitoringLocationIdentifier": "string",
            "site_no": "string",
            "huc_cd": "string",
        },
    )
    for col in ["dec_long_va", "dec_lat_va", "D50"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["huc2"] = huc2_from_huc_cd(data["huc_cd"])
    return data.dropna(subset=["dec_long_va", "dec_lat_va", "D50"]).copy()


def station_summary(data: pd.DataFrame) -> pd.DataFrame:
    station = (
        data.groupby("MonitoringLocationIdentifier", dropna=False)
        .agg(
            station_nm=("station_nm", "first"),
            site_no=("site_no", "first"),
            dec_long_va=("dec_long_va", "first"),
            dec_lat_va=("dec_lat_va", "first"),
            huc2=("huc2", "first"),
            n_samples=("D50", "size"),
            median_d50=("D50", "median"),
        )
        .reset_index()
    )
    station["grain_class"] = station["median_d50"].map(grain_class)
    return station[station["grain_class"].isin(GRAIN_ORDER)].copy()


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


def stations_to_wbd(stations: pd.DataFrame, wbd):
    import geopandas as gpd

    points = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations.dec_long_va, stations.dec_lat_va),
        crs="EPSG:4269",
    ).to_crs(CRS)
    boundary_huc = wbd[["huc2", "geometry"]].rename(columns={"huc2": "huc2_wbd"})
    joined = gpd.sjoin(points, boundary_huc, how="left", predicate="intersects")
    joined["huc2"] = (
        joined["huc2_wbd"]
        .fillna(joined["huc2"])
        .astype("string")
        .str.zfill(2)
    )
    return joined


def plot_map_region(
    ax: plt.Axes,
    wbd,
    stations,
    huc2_values: list[str],
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    label_huc: bool = False,
    left_pad: float = 0.0,
    point_size: float = 13.5,
) -> None:
    wbd_sub = wbd[wbd["huc2"].isin(huc2_values)].copy()
    stations_sub = stations[stations["huc2"].isin(huc2_values)].copy()

    wbd_sub.plot(
        ax=ax,
        color=WBD_FACE_COLOR,
        edgecolor=BOUNDARY_COLOR,
        linewidth=0.28,
        alpha=1.0,
        zorder=1,
    )
    for grain in GRAIN_ORDER:
        subset = stations_sub[stations_sub["grain_class"] == grain]
        if subset.empty:
            continue
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            c=subset["n_samples"],
            cmap=cmap,
            norm=norm,
            marker=GRAIN_MARKERS[grain],
            s=point_size,
            linewidths=0.28,
            edgecolors="#2F2F2F",
            alpha=0.92,
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


def plot_map_fallback(
    ax: plt.Axes,
    stations: pd.DataFrame,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
) -> None:
    conus = stations[
        stations["dec_long_va"].between(-126, -66)
        & stations["dec_lat_va"].between(24, 50)
    ]
    for grain in GRAIN_ORDER:
        subset = conus[conus["grain_class"] == grain]
        if subset.empty:
            continue
        ax.scatter(
            subset["dec_long_va"],
            subset["dec_lat_va"],
            c=subset["n_samples"],
            cmap=cmap,
            norm=norm,
            marker=GRAIN_MARKERS[grain],
            s=13,
            linewidths=0.28,
            edgecolors="#2F2F2F",
            alpha=0.92,
            zorder=3,
        )
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def add_map_legends(
    ax: plt.Axes,
    fig: plt.Figure,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
) -> None:
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=GRAIN_MARKERS[grain],
            color="none",
            markerfacecolor="white",
            markeredgecolor="#2F2F2F",
            markeredgewidth=0.75,
            markersize=4.2,
            label=GRAIN_LABELS[grain].replace("\n", " "),
        )
        for grain in GRAIN_ORDER
    ]
    ax.legend(
        handles=shape_handles,
        title=r"Station median $D_{50}$",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.5, 0.9),
        ncol=2,
        title_fontsize=6.0,
        handletextpad=1,
        labelspacing=1,
        columnspacing=1,
        borderaxespad=0.15,
        fontsize=5.4,
    )

    cax = ax.inset_axes([0.36, -0.05, 0.50, 0.032])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks(SAMPLE_TICKS)
    cbar.set_ticklabels([str(tick) for tick in SAMPLE_TICKS])
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.set_label("Number of samples per station", labelpad=2)
    cbar.outline.set_linewidth(0.35)
    cbar.ax.tick_params(labelsize=5.4, width=0.45, length=2)


def add_scale_bar(ax: plt.Axes) -> None:
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
        return


def plot_map(ax: plt.Axes, fig: plt.Figure, stations: pd.DataFrame, wbd) -> None:
    max_samples = max(int(stations["n_samples"].max()), 750)
    norm = mpl.colors.LogNorm(vmin=1, vmax=max_samples)
    cmap = sample_count_cmap()

    if wbd is None:
        plot_map_fallback(ax, stations, norm, cmap)
    else:
        joined = stations_to_wbd(stations, wbd)
        conus = [f"0{i}" for i in range(1, 10)] + [str(i) for i in range(10, 19)]
        plot_map_region(
            ax,
            wbd,
            joined,
            conus,
            norm,
            cmap,
            label_huc=True,
            left_pad=0.30,
            point_size=13.5,
        )

        inset_specs = [
            ("19", [0.01, 0.60, 0.22, 0.36], "AK"),
            ("20", [-0.03, 0.22, 0.40, 0.44], "HI"),
            ("21", [0.20, 0.03, 0.17, 0.12], "PR"),
        ]
        for huc2, rect, label in inset_specs:
            iax = ax.inset_axes(rect)
            plot_map_region(
                iax,
                wbd,
                joined,
                [huc2],
                norm,
                cmap,
                label_huc=True,
                point_size=11.0,
            )
            iax.text(0.03, 0.05, label, transform=iax.transAxes, fontsize=5.2)
            for spine in iax.spines.values():
                spine.set_visible(True)
                spine.set_color("#D0D0D0")
                spine.set_linewidth(0.55)
        add_scale_bar(ax)

    add_map_legends(ax, fig, norm, cmap)


def plot_sample_count_hist(ax: plt.Axes, stations: pd.DataFrame) -> None:
    counts, edges = np.histogram(stations["n_samples"], bins=SAMPLE_COUNT_EDGES)
    ax.stairs(counts, edges, color=SAMPLE_LINE_COLOR, linewidth=1.35)
    ax.set_xscale("log")
    ax.set_xlim(SAMPLE_COUNT_EDGES[0], SAMPLE_COUNT_EDGES[-1])
    ax.set_xticks(SAMPLE_TICKS)
    ax.set_xticklabels(["1", "2", "5", "10", "25", "50", "100", "250", "750"])
    ax.set_xlabel("Number of samples per station")
    ax.set_ylabel("Number of stations")
    style_axis(ax)
    one_sample = int((stations["n_samples"] == 1).sum())
    pct = one_sample / len(stations) * 100
    ax.text(
        0.98,
        0.92,
        f"{pct:.1f}% one-sample stations\nmax = {int(stations['n_samples'].max())}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color=AXIS_COLOR,
    )


def plot_d50_hist(ax: plt.Axes, samples: pd.DataFrame) -> None:
    d50 = samples["D50"].to_numpy(dtype=float)
    counts, edges = np.histogram(d50[np.isfinite(d50)], bins=D50_EDGES)
    ax.stairs(counts, edges, color=D50_LINE_COLOR, linewidth=1.35)
    for boundary in [2, 4, 64]:
        ax.axvline(boundary, color="#B9B9B9", linewidth=0.6, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(D50_EDGES[0], D50_EDGES[-1])
    ax.set_xticks([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(
        ["0.063", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64", "128"],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel(r"$D_{50}$ (mm)")
    ax.set_ylabel("Number of samples")
    style_axis(ax)
    ax.text(
        0.98,
        0.92,
        f"n = {len(samples):,} samples\nmedian = {np.nanmedian(d50):.2f} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color=AXIS_COLOR,
    )


def make_figure(data_path: Path, wbd_path: Path | None, out_prefix: Path,
                fmt: str = "png") -> None:
    setup_style()
    samples = load_samples(data_path)
    stations = station_summary(samples)
    wbd = load_wbd(wbd_path)

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[2, 1.0],
        left=0.065,
        right=0.985,
        top=0.965,
        bottom=0.125,
        hspace=0.30,
        wspace=0.22,
    )
    ax_map = fig.add_subplot(gs[0, :])
    ax_samples = fig.add_subplot(gs[1, 0])
    ax_d50 = fig.add_subplot(gs[1, 1])

    plot_map(ax_map, fig, stations, wbd)
    plot_sample_count_hist(ax_samples, stations)
    plot_d50_hist(ax_d50, samples)
    add_figure_panel_labels(fig, ax_samples, ax_d50)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    save_figure(fig, png_path.with_suffix(""), fmt, dpi=600, facecolor="white", bbox_inches=None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft Figure S2: observed PSD station coverage and D50 distributions."
    )
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to DATA_stats.csv")
    parser.add_argument("--wbd", default=str(WBD_PATH), help="Path to the WBDHU2 GeoPackage (data/reference/wbd_hu2.gpkg)")
    parser.add_argument("--no-wbd", action="store_true", help="Use lon/lat station scatter instead of WBD map")
    parser.add_argument("--out", default=str(OUT_PREFIX), help="Output prefix, without extension")
    add_format_argument(parser)
    args = parser.parse_args()

    make_figure(
        data_path=Path(args.data).expanduser(),
        wbd_path=None if args.no_wbd else Path(args.wbd).expanduser(),
        out_prefix=Path(args.out).expanduser(),
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
