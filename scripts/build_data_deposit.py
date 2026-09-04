#!/usr/bin/env python3
"""Assemble the Zenodo data deposit.

``--phase archive`` copies the inputs and the fitted parameters out of the
analysis tree given by ``--source``. ``--phase derived`` copies the tables this
repository regenerates. ``--phase all`` does both and writes ``manifest.csv``.

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

PERCENTILES = (5, 16, 25, 50, 75, 84, 95)
FUNCTIONS = ("GIG_2p", "Lognormal", "Weibull")

SAMPLE_ID_COLS = ["MonitoringLocationIdentifier", "site_no", "sample_ID"]
ACTIVITY_COLS = [
    "ActivityStartDate", "ActivityStartTime/Time", "ActivityStartTime/TimeZoneCode",
    "HydrologicCondition", "HydrologicEvent",
]
STATION_COLS = [
    "station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va", "coord_acy_cd",
    "dec_coord_datum_cd", "huc_cd", "drain_area_va", "contrib_drain_area_va",
]
STAT_COLS = [
    "num_dp", "D50", "d50_grainsize", "gsc_silt", "gsc_sand", "gsc_granule",
    "gsc_pebble", "gsc_cobble", "grainclass_more50", "mean_size_FW", "stand_div_FW",
    "skew_FW", "kurtosis_FW", "Sigma_g", "sample_ST_type", "stream_type",
    "best_function(s)", "num_functions",
]
PRINCIPAL_CODES = [f"p{code}" for code in range(80164, 80176)]

# Supplementary Table S1: full name and the literature the form is drawn from.
FUNCTION_NAMES = {
    1: ("Algebraic", "math/stats"), 2: ("Error Power", "newly introduced"),
    3: ("Exponential Power", "pedology"), 4: ("Gamma", "math/stats"),
    5: ("Generalized Log-Hyperbolic, special case p = -1/2", "sediment"),
    6: ("Generalized Log-Hyperbolic, special case p = 1", "sediment"),
    7: ("Hyperbolic Tangent", "sediment"), 8: ("Logarithm exponential", "pedology"),
    9: ("Linear logarithm", "pedology"), 10: ("Lognormal", "sediment"),
    11: ("Log-Laplace", "math/stats"), 12: ("Normal Inverse Gaussian", "sediment"),
    13: ("Power Law", "pedology"),
    14: ("Two-parameter Generalized Inverse Gaussian", "math/stats; hydrology"),
    15: ("Two-parameter Weibull", "sediment"),
    16: ("Generalized Log-Hyperbolic", "sediment"),
    17: ("Lognormal and two-parameter Weibull combination", "newly introduced"),
    18: ("Lognormal and Power Law combination", "newly introduced"),
    19: ("Lognormal and Hyperbolic Tangent combination", "newly introduced"),
    20: ("Log-skew-Laplace", "sediment"),
    21: ("Power Law and Hyperbolic Tangent combination", "newly introduced"),
    22: ("Power Law Exponential, first alternative", "pedology"),
    23: ("Power Law Exponential, second alternative", "pedology"),
    24: ("Power Law and two-parameter Weibull combination", "newly introduced"),
    25: ("Three-parameter Generalized Inverse Gaussian", "math/stats; hydrology"),
}

RHINE_NOTICE = """\
# Lower Rhine data are not redistributed here

The independent validation samples come from:

> Chowdhury, K., Blom, A., Ylla Arbos, C. & Schielen, R. M. J. *Schematized model of the
> Lower Rhine River and its branches in SOBEK RE*, version 2. 4TU.ResearchData (2025).
> https://doi.org/10.4121/eb78267a-137b-4f61-bb7e-6549915a24c7

That dataset is published under **CC BY-NC-ND 4.0**, which permits neither
redistribution nor derivative works. We therefore cannot include the samples, or
the per-sample fitted parameters derived from them, in this deposit.

To reproduce the Lower Rhine results (Supplementary Fig. S5 and the Rhine
paragraph of the main text):

1. Download the dataset from the DOI above and accept its licence.
2. Extract the bed-sediment gradation table for the Lower Rhine and its branches
   into a CSV with one row per sample and one column per sieve size, each column
   named for its size in millimetres:

       river_km,river_name,site_no,0.5 mm,2 mm,8 mm,31.5 mm,125 mm,sample_ID
       849,Bovenrijn-Waal,Bovenrijn-Waal_849,8.61,19.4,41.77,92.81,100.0,rhine_1

3. Save it in the code repository as `data/rhine/rhine_data_to_fit.csv`.
4. Run `make rhine`.

Aggregate Rhine results as published in the paper are in `06_tables/`.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_percentiles(source: Path, final: pd.DataFrame, deposit: Path) -> None:
    """One percentile table, assembled from the analysis tree.

    Percentiles are solved from the fitted CDFs in the original analysis; they are
    carried through unchanged rather than re-derived. `_reference` is the mean over
    the functions in each sample's best-fit set.
    """
    target = deposit / "03_percentiles"
    target.mkdir(parents=True, exist_ok=True)

    table = final[["sample_ID", "site_no", "best_function(s)", "num_functions",
                   "skew_FW", "stream_type", "sample_ST_type"]].copy()
    table["sample_ID"] = table["sample_ID"].astype(int)
    index = {name: i for i, name in enumerate(FUNCTIONS)}

    for percentile in PERCENTILES:
        label = f"D{percentile:02d}"
        legacy = pd.read_csv(source.parent / "psd" / "Dvalues" / f"{label}_values.csv")
        legacy["sample_ID"] = legacy["sample_ID"].astype(int)
        table = table.merge(legacy[["sample_ID"] + [f"{label}_{n}" for n in FUNCTIONS]],
                            on="sample_ID", how="left")
        matrix = table[[f"{label}_{n}" for n in FUNCTIONS]].to_numpy(float)
        reference = np.full(len(table), np.nan)
        for row, best in enumerate(table["best_function(s)"]):
            picks = [index[p.strip()] for p in str(best).split("&") if p.strip() in index]
            if picks and np.isfinite(matrix[row, picks]).any():
                reference[row] = np.nanmean(matrix[row, picks])
        table[f"{label}_reference"] = reference

    ordered = ["sample_ID", "site_no", "best_function(s)", "num_functions", "skew_FW",
               "stream_type", "sample_ST_type"]
    for percentile in PERCENTILES:
        label = f"D{percentile:02d}"
        ordered += [f"{label}_reference"] + [f"{label}_{n}" for n in FUNCTIONS]
    table[ordered].to_csv(target / "dvalues.csv", index=False)
    print(f"  03_percentiles/dvalues.csv       {len(table):,} rows x {len(ordered)} cols")


def phase_archive(source: Path, deposit: Path) -> None:
    samples_dir = deposit / "01_samples"
    fits_dir = deposit / "02_fits"
    hydro_dir = deposit / "04_hydraulics"
    for directory in (samples_dir, fits_dir, hydro_dir, deposit / "05_rhine"):
        directory.mkdir(parents=True, exist_ok=True)

    print("archive:")
    final = pd.read_csv(source / "output_data" / "DATA_final_all_optfuncs.csv",
                        dtype={"site_no": str}, low_memory=False)
    codes = [c for c in final.columns if c.startswith("p") and c[1:].isdigit()]

    samples = final[SAMPLE_ID_COLS + ACTIVITY_COLS + ["num_dp"] + codes]
    samples.to_csv(samples_dir / "usgs_psd_samples_qc.csv", index=False)
    print(f"  usgs_psd_samples_qc.csv          {len(samples):,} rows x {len(samples.columns)} cols")

    stats = final[["sample_ID", "site_no"] + STAT_COLS].copy()
    single = pd.read_csv(source / "output_data" / "BIC_one_best_2p_func_for_each_sample.csv",
                         dtype={"site_no": str})
    stats = stats.merge(
        single[["sample_ID", "Function"]].rename(columns={"Function": "best_of_three_by_BIC"}),
        on="sample_ID", how="left")
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
    sizes.sort_values("size_mm").to_csv(samples_dir / "sieve_code_sizes.csv", index=False)
    print(f"  sieve_code_sizes.csv             {len(sizes)} grades")

    catalog = []
    for spec in FUNCTION_SPECS:
        matches = sorted((source / "fitted_functions").glob(f"F{spec.number:02d}_*.csv"))
        if not matches:
            raise FileNotFoundError(f"No fit file for F{spec.number:02d} {spec.label}")
        fit = pd.read_csv(matches[0], dtype={"site_no": str})
        fit["sample_ID"] = fit["sample_ID"].astype(int)
        keep = ["site_no", "sample_ID"]
        keep += [c for c in ("fitted_A", "fitted_B", "fitted_C") if c in fit.columns]
        keep += ["RMSE", "R^2", "AIC", "BIC"]
        fit[keep].rename(columns={"R^2": "R2"}).to_csv(
            fits_dir / f"fits_F{spec.number:02d}_{spec.label}.csv", index=False)
        full_name, source_field = FUNCTION_NAMES[spec.number]
        catalog.append({
            "number": spec.number, "abbreviation": spec.label, "full_name": full_name,
            "source_field": source_field, "n_parameters": spec.n_params,
            "grid_searched": spec.grid is not None,
            "n_initial_guesses": len(spec.grid.initial_values()) if spec.grid else 2,
            "fit_x_transform": spec.base_x_transform,
            "implementation": f"psd_gig.function_library.{spec.func.__name__}",
            "file": f"fits_F{spec.number:02d}_{spec.label}.csv",
        })
    pd.DataFrame(catalog).to_csv(fits_dir / "function_catalog.csv", index=False)
    print(f"  fits_F01..F25 + function_catalog.csv")

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

    build_percentiles(source, final, deposit)
    (deposit / "05_rhine" / "README.md").write_text(RHINE_NOTICE, encoding="utf-8")
    print("  05_rhine/README.md               licence notice")


DERIVED = [
    ("outputs/water/sample_water_metrics.csv", "04_hydraulics"),
    ("outputs/tables/table_S2_median_metrics.csv", "06_tables"),
    ("outputs/tables/table_S3_best_counts.csv", "06_tables"),
    ("outputs/tables/table_S4_by_n_parameters.csv", "06_tables"),
    ("outputs/tables/table_S5_significance.csv", "06_tables"),
    ("outputs/tables/figure4_delta_bic_summary.csv", "06_tables"),
    ("outputs/tables/figure5_rmsre_matrix.csv", "06_tables"),
    ("outputs/rhine/rhine_function_summary.csv", "06_tables"),
    ("outputs/rhine/rhine_best_fit_by_skew_class.csv", "06_tables"),
]


def phase_derived(deposit: Path) -> None:
    print("derived:")
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
        if not path.is_file() or path.name in {"manifest.csv", ".DS_Store"}:
            continue
        n_rows = n_cols = ""
        if path.suffix == ".csv":
            frame = pd.read_csv(path, low_memory=False)
            n_rows, n_cols = len(frame), len(frame.columns)
        rows.append({"path": str(path.relative_to(deposit)), "bytes": path.stat().st_size,
                     "rows": n_rows, "columns": n_cols, "sha256": sha256(path)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(deposit / "manifest.csv", index=False)
    print(f"\nmanifest.csv: {len(manifest)} files, {manifest['bytes'].sum() / 1e6:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["archive", "derived", "all"], default="all")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
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
