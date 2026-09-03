#!/usr/bin/env python3
"""Assemble the Zenodo data deposit.

Two phases:

``--phase archive``
    Copies and normalises the archived inputs and fitted parameters out of the
    legacy analysis tree (``--source``). These are the artefacts that produced
    the published results; they are not regenerated here.

``--phase derived``
    Copies the tables this repository regenerates (percentiles, water-security
    metrics, supplementary tables and figure source data) out of ``outputs/``.

``--phase all`` runs both, then writes ``manifest.csv``.

    python scripts/build_data_deposit.py --phase all
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psd_gig.fit_specs import FUNCTION_SPECS  # noqa: E402

DEFAULT_SOURCE = ROOT.parent.parent / "Lingbo"
DEFAULT_DEPOSIT = ROOT.parent / "psd-gig-data-v1.0.0"

SAMPLE_ID_COLS = ["MonitoringLocationIdentifier", "site_no", "sample_ID"]
ACTIVITY_COLS = [
    "ActivityStartDate",
    "ActivityStartTime/Time",
    "ActivityStartTime/TimeZoneCode",
    "HydrologicCondition",
    "HydrologicEvent",
]
STATION_COLS = [
    "station_nm",
    "site_tp_cd",
    "dec_lat_va",
    "dec_long_va",
    "coord_acy_cd",
    "dec_coord_datum_cd",
    "huc_cd",
    "drain_area_va",
    "contrib_drain_area_va",
]
STAT_COLS = [
    "num_dp",
    "D50",
    "d50_grainsize",
    "gsc_silt",
    "gsc_sand",
    "gsc_granule",
    "gsc_pebble",
    "gsc_cobble",
    "grainclass_more50",
    "mean_size_FW",
    "stand_div_FW",
    "skew_FW",
    "kurtosis_FW",
    "Sigma_g",
    "sample_ST_type",
    "stream_type",
    "best_function(s)",
    "num_functions",
]
PRINCIPAL_CODES = [f"p{code}" for code in range(80164, 80176)]

# Supplementary Table S1: full name and the literature the form is drawn from.
FUNCTION_NAMES = {1: ('Algebraic', 'math/stats'), 2: ('Error Power', 'newly introduced'), 3: ('Exponential Power', 'pedology'), 4: ('Gamma', 'math/stats'), 5: ('Generalized Log-Hyperbolic, special case p = -1/2', 'sediment'), 6: ('Generalized Log-Hyperbolic, special case p = 1', 'sediment'), 7: ('Hyperbolic Tangent', 'sediment'), 8: ('Logarithm exponential', 'pedology'), 9: ('Linear logarithm', 'pedology'), 10: ('Lognormal', 'sediment'), 11: ('Log-Laplace', 'math/stats'), 12: ('Normal Inverse Gaussian', 'sediment'), 13: ('Power Law', 'pedology'), 14: ('Two-parameter Generalized Inverse Gaussian', 'math/stats; hydrology'), 15: ('Two-parameter Weibull', 'sediment'), 16: ('Generalized Log-Hyperbolic', 'sediment'), 17: ('Lognormal and two-parameter Weibull combination', 'newly introduced'), 18: ('Lognormal and Power Law combination', 'newly introduced'), 19: ('Lognormal and Hyperbolic Tangent combination', 'newly introduced'), 20: ('Log-skew-Laplace', 'sediment'), 21: ('Power Law and Hyperbolic Tangent combination', 'newly introduced'), 22: ('Power Law Exponential, first alternative', 'pedology'), 23: ('Power Law Exponential, second alternative', 'pedology'), 24: ('Power Law and two-parameter Weibull combination', 'newly introduced'), 25: ('Three-parameter Generalized Inverse Gaussian', 'math/stats; hydrology')}

RHINE_NOTICE = """\
# Lower Rhine data are not redistributed here

The independent validation samples come from:

> Chowdhury, K., Blom, A., Ylla Arbos, C. & Schielen, R. M. J. *Schematized model of the
> Lower Rhine River and its branches in SOBEK RE*, version 2. 4TU.ResearchData (2025).
> https://doi.org/10.4121/eb78267a-137b-4f61-bb7e-6549915a24c7

That dataset is published under **CC BY-NC-ND 4.0**, which permits neither
redistribution of derivative works nor commercial reuse. We therefore cannot
include the samples, or the per-sample fitted parameters derived from them, in
this deposit.

To reproduce the Lower Rhine results (Supplementary Fig. S5 and the Rhine
paragraph of the main text):

1. Download the dataset from the DOI above and accept its licence.
2. Extract the reach-averaged bed-sediment gradation table for the Lower Rhine
   and its branches into a CSV with one row per sample and one column per sieve
   size, the column named for the size in millimetres, for example:

       river_km,river_name,site_no,0.5 mm,2 mm,8 mm,31.5 mm,125 mm,sample_ID
       849,Bovenrijn-Waal,Bovenrijn-Waal_849,8.61,19.4,41.77,92.81,100.0,rhine_1

3. Save it in the code repository as `data/rhine/rhine_data_to_fit.csv`.
4. Run `make rhine`.

Aggregate Rhine results as published in the paper -- median BIC and RMSE per
function, and best-fit counts by skewness class -- are included in
`06_tables_and_figure_data/` so that the figure and the quoted numbers remain
checkable without the restricted source data.
"""


ARCHIVED_PERCENTILES = (5, 16, 25, 50, 75, 84, 95)


def build_archived_percentiles(source: Path, final: pd.DataFrame, deposit: Path) -> None:
    """Assemble the percentile table that the published Fig. 5 numbers were computed from.

    These come from the original analysis, where the GIG inverse was evaluated by a
    grid scan rather than analytically. `scripts/02_compute_dvalues.py` regenerates
    them from the fitted parameters; the two agree to ~1e-4 relative. See
    docs/caveats.md -- the archived table is the default input to the water-metric
    step so that the published numbers reproduce exactly.
    """
    target = deposit / "03_percentiles"
    target.mkdir(parents=True, exist_ok=True)
    functions = ("GIG_2p", "Lognormal", "Weibull")

    table = final[["sample_ID", "site_no", "best_function(s)", "num_functions",
                   "skew_FW", "stream_type", "sample_ST_type"]].copy()
    table["sample_ID"] = table["sample_ID"].astype(int)
    index = {name: i for i, name in enumerate(functions)}

    for percentile in ARCHIVED_PERCENTILES:
        label = f"D{percentile:02d}"
        legacy = pd.read_csv(source.parent / "psd" / "Dvalues" / f"D{percentile:02d}_values.csv")
        legacy["sample_ID"] = legacy["sample_ID"].astype(int)
        columns = ["sample_ID"] + [f"{label}_{name}" for name in functions]
        table = table.merge(legacy[columns], on="sample_ID", how="left")
        matrix = table[[f"{label}_{name}" for name in functions]].to_numpy(float)
        reference = np.full(len(table), np.nan)
        for row_index, best in enumerate(table["best_function(s)"]):
            picks = [index[part.strip()] for part in str(best).split("&")
                     if part.strip() in index]
            if picks:
                values = matrix[row_index, picks]
                if np.isfinite(values).any():
                    reference[row_index] = np.nanmean(values)
        table[f"{label}_reference"] = reference

    ordered = ["sample_ID", "site_no", "best_function(s)", "num_functions", "skew_FW",
               "stream_type", "sample_ST_type"]
    for percentile in ARCHIVED_PERCENTILES:
        label = f"D{percentile:02d}"
        ordered += [f"{label}_reference"] + [f"{label}_{name}" for name in functions]
    table[ordered].to_csv(target / "dvalues_archived.csv", index=False)
    print(f"  03_percentiles/dvalues_archived.csv  {len(table):,} rows x "
          f"{len(ordered)} cols")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def size_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("p") and c[1:].isdigit()]


def phase_archive(source: Path, deposit: Path) -> None:
    samples_dir = deposit / "01_samples"
    fits_dir = deposit / "02_fits"
    hydro_dir = deposit / "04_hydraulics"
    for directory in (samples_dir, fits_dir, hydro_dir, deposit / "05_rhine"):
        directory.mkdir(parents=True, exist_ok=True)

    print("archive: reading legacy tables")
    final = pd.read_csv(source / "output_data" / "DATA_final_all_optfuncs.csv",
                        dtype={"site_no": str}, low_memory=False)
    codes = size_columns(final)

    samples = final[SAMPLE_ID_COLS + ACTIVITY_COLS + ["num_dp"] + codes].copy()
    samples.to_csv(samples_dir / "usgs_psd_samples_qc.csv", index=False)
    print(f"  usgs_psd_samples_qc.csv          {len(samples):,} rows x {len(samples.columns)} cols")

    stats = final[["sample_ID", "site_no"] + STAT_COLS].copy()
    stats.to_csv(samples_dir / "usgs_sample_statistics.csv", index=False)
    print(f"  usgs_sample_statistics.csv       {len(stats):,} rows")

    stations = final[["site_no"] + STATION_COLS + ["stream_type"]].drop_duplicates("site_no")
    counts = final.groupby("site_no").size().rename("num_samples")
    stations = stations.merge(counts, on="site_no", how="left").sort_values("site_no")
    stations.to_csv(samples_dir / "usgs_station_metadata.csv", index=False)
    print(f"  usgs_station_metadata.csv        {len(stations):,} rows")

    lookup = pd.read_csv(source / "sediment_data" / "pamcodes_size.csv")
    sizes = pd.DataFrame({
        "parameter_code": [c.lstrip("p") for c in lookup.columns],
        "column_name": list(lookup.columns),
        "size_mm": lookup.iloc[0].astype(float).to_numpy(),
    })
    sizes["principal_grade"] = sizes["column_name"].isin(PRINCIPAL_CODES)
    sizes = sizes.sort_values("size_mm")
    sizes.to_csv(samples_dir / "sieve_code_sizes.csv", index=False)
    print(f"  sieve_code_sizes.csv             {len(sizes)} grades")

    print("archive: fitted parameters")
    catalog = []
    for spec in FUNCTION_SPECS:
        matches = sorted((source / "fitted_functions").glob(f"F{spec.number:02d}_*.csv"))
        if not matches:
            raise FileNotFoundError(f"No archived fit file for F{spec.number:02d} {spec.label}")
        fit = pd.read_csv(matches[0], dtype={"site_no": str})
        fit["sample_ID"] = fit["sample_ID"].astype(int)
        keep = ["site_no", "sample_ID"]
        keep += [c for c in ("fitted_A", "fitted_B", "fitted_C") if c in fit.columns]
        keep += ["RMSE", "R^2", "AIC", "BIC"]
        fit = fit[keep].rename(columns={"R^2": "R2"})
        fit.to_csv(fits_dir / f"fits_F{spec.number:02d}_{spec.label}.csv", index=False)
        full_name, source_field = FUNCTION_NAMES[spec.number]
        catalog.append({
            "number": spec.number,
            "abbreviation": spec.label,
            "full_name": full_name,
            "source_field": source_field,
            "n_parameters": spec.n_params,
            "grid_searched": spec.grid is not None,
            "n_initial_guesses": len(spec.grid.initial_values()) if spec.grid else 2,
            "fit_x_transform": spec.base_x_transform,
            "implementation": f"psd_gig.function_library.{spec.func.__name__}",
            "archived_file": f"fits_F{spec.number:02d}_{spec.label}.csv",
            "fit_layer": "grid-selected" if "grids_well" in matches[0].name else "direct",
            "legacy_file": matches[0].name,
        })
    pd.DataFrame(catalog).to_csv(fits_dir / "function_catalog.csv", index=False)
    print(f"  fits_F01..F25 + function_catalog.csv ({len(catalog)} functions)")

    # Six of the nine grid-searched functions also have a single-initial-guess fit in
    # the legacy tree. Those are what the manuscript's main text quotes for median BIC,
    # while Table S2 quotes the grid-selected values, so both are deposited.
    base_dir = fits_dir / "base_fits"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_written = []
    for spec in FUNCTION_SPECS:
        candidates = [p for p in (source / "fitted_functions").glob("f*.csv")
                      if p.stem.lower() == f"f{spec.number:02d}_{spec.label}".lower()]
        if not candidates:
            continue
        fit = pd.read_csv(candidates[0], dtype={"site_no": str})
        fit["sample_ID"] = fit["sample_ID"].astype(int)
        keep = ["site_no", "sample_ID"]
        keep += [c for c in ("fitted_A", "fitted_B", "fitted_C") if c in fit.columns]
        keep += ["RMSE", "R^2", "AIC", "BIC"]
        fit[keep].rename(columns={"R^2": "R2"}).to_csv(
            base_dir / f"base_F{spec.number:02d}_{spec.label}.csv", index=False)
        base_written.append(spec.label)
    print(f"  base_fits/ ({len(base_written)}): {', '.join(base_written)}")

    best = final[["site_no", "sample_ID", "best_function(s)", "num_functions"]].copy()
    single = pd.read_csv(source / "output_data" / "BIC_one_best_2p_func_for_each_sample.csv",
                         dtype={"site_no": str})
    best = best.merge(
        single[["sample_ID", "Function", "BIC"]].rename(
            columns={"Function": "best_of_three_by_BIC", "BIC": "best_of_three_BIC"}),
        on="sample_ID", how="left")
    best.to_csv(fits_dir / "best_fit_by_sample.csv", index=False)
    print(f"  best_fit_by_sample.csv           {len(best):,} rows")

    print("archive: station hydraulics")
    stations_comid = pd.read_csv(source / "water_security" / "stations_comid_hyriver.csv",
                                 dtype={"site_no": str, "comid": "Int64"})
    stations_comid = stations_comid.rename(columns={"comid": "COMID"}).drop_duplicates("site_no")
    attrs = pd.read_parquet(
        source / "water_security" / "NHD_SB_global_data.parquet",
        columns=["COMID", "SLOPE", "BANKFULL_XSEC_AREA", "BANKFULL_WIDTH",
                 "BANKFULL_DEPTH", "QA_MA", "VA_MA"],
    ).drop_duplicates("COMID")
    nhd = stations_comid.merge(attrs, on="COMID", how="left")
    nhd = nhd[nhd["site_no"].isin(set(final["site_no"]))].sort_values("site_no")
    nhd.to_csv(hydro_dir / "station_nhdplus_attributes.csv", index=False)
    print(f"  station_nhdplus_attributes.csv   {len(nhd):,} rows")

    q100 = pd.read_csv(source / "output_data" / "hydro_q100_cache.csv", dtype={"site_no": str})
    q100.to_csv(hydro_dir / "station_q100_logpearson3.csv", index=False)
    print(f"  station_q100_logpearson3.csv     {len(q100):,} rows")

    build_archived_percentiles(source, final, deposit)

    (deposit / "05_rhine" / "README.md").write_text(RHINE_NOTICE, encoding="utf-8")
    print("  05_rhine/README.md               licence notice")


DERIVED = [
    ("outputs/dvalues/dvalues_recomputed.csv", "03_percentiles"),
    ("outputs/dvalues/dvalues_archived_vs_recomputed.csv", "03_percentiles"),
    ("outputs/water/sample_water_metrics.csv", "04_hydraulics"),
    ("outputs/tables/table_S2_median_metrics.csv", "06_tables_and_figure_data"),
    ("outputs/tables/table_S3_best_counts.csv", "06_tables_and_figure_data"),
    ("outputs/tables/table_S4_by_n_parameters.csv", "06_tables_and_figure_data"),
    ("outputs/tables/table_S5_significance.csv", "06_tables_and_figure_data"),
    ("outputs/tables/figure5_rmsre_matrix.csv", "06_tables_and_figure_data"),
    ("outputs/tables/figure4_delta_bic_summary.csv", "06_tables_and_figure_data"),
    ("outputs/tables/cdf_validity.csv", "06_tables_and_figure_data"),
    ("outputs/rhine/rhine_function_summary.csv", "06_tables_and_figure_data"),
    ("outputs/rhine/rhine_best_fit_by_skew_class.csv", "06_tables_and_figure_data"),
]


def phase_derived(deposit: Path) -> None:
    print("derived: copying regenerated tables")
    for relative, group in DERIVED:
        source_path = ROOT / relative
        if not source_path.exists():
            print(f"  MISSING {relative} -- run the pipeline first")
            continue
        target_dir = deposit / group
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_dir / source_path.name)
        print(f"  {group}/{source_path.name}")


def write_manifest(deposit: Path) -> None:
    rows = []
    for path in sorted(deposit.rglob("*")):
        if not path.is_file() or path.name == "manifest.csv":
            continue
        relative = path.relative_to(deposit)
        n_rows = ""
        n_cols = ""
        if path.suffix == ".csv":
            frame = pd.read_csv(path, low_memory=False)
            n_rows, n_cols = len(frame), len(frame.columns)
        rows.append({
            "path": str(relative),
            "bytes": path.stat().st_size,
            "rows": n_rows,
            "columns": n_cols,
            "sha256": sha256(path),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(deposit / "manifest.csv", index=False)
    total = manifest["bytes"].sum()
    print(f"\nmanifest.csv: {len(manifest)} files, {total / 1e6:.1f} MB total")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["archive", "derived", "all"], default="all")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Legacy analysis tree holding the archived inputs and fits.")
    parser.add_argument("--deposit", type=Path, default=DEFAULT_DEPOSIT)
    args = parser.parse_args()

    args.deposit.mkdir(parents=True, exist_ok=True)
    if args.phase in {"archive", "all"}:
        phase_archive(args.source, args.deposit)
    if args.phase in {"derived", "all"}:
        phase_derived(args.deposit)
    write_manifest(args.deposit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
