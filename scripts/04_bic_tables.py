#!/usr/bin/env python3
"""Supplementary Tables S2-S4, and the Fig. 4 source data, from the fitted parameters.

Table S2  median BIC, RMSE and R2 per function, plus the count of upper BIC
          outliers (Tukey rule: BIC > Q3 + 1.5 IQR).
Table S3  how often each function is best, at three tolerances: strictly the
          lowest BIC, and within 2 or 6 BIC units of the sample's minimum.
Table S4  the same counts computed separately within the two- and
          three-parameter families.
Figure 4  per-function delta-BIC quantiles relative to each sample's best
          function, and the share of samples where each function wins outright.

It also writes three intermediates the figure scripts consume: the BIC and RMSE
matrices (one row per sample, one column per function) and the joined sample /
station analysis frame.

    python scripts/04_bic_tables.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psd_gig.fit_specs import FUNCTION_SPECS  # noqa: E402

TOLERANCES = (2, 6)


def load_metrics(fit_dir: Path, keep_site: bool = False) -> dict[str, pd.DataFrame]:
    columns = ["sample_ID", "RMSE", "R2", "BIC"] + (["site_no"] if keep_site else [])
    frames = {}
    for spec in FUNCTION_SPECS:
        path = fit_dir / f"fits_F{spec.number:02d}_{spec.label}.csv"
        frame = pd.read_csv(path, usecols=columns, dtype={"site_no": str})
        frames[spec.label] = frame.drop_duplicates("sample_ID").set_index("sample_ID")
    return frames


def write_matrices(frames: dict[str, pd.DataFrame], labels: list[str], out: Path) -> None:
    """Sample x function matrices of BIC and RMSE, consumed by fig_04 and fig_S03."""
    site = next(iter(frames.values()))["site_no"]
    for column, name in (("BIC", "bic_matrix.tsv"), ("RMSE", "rmse_matrix.tsv")):
        table = pd.DataFrame({label: frames[label][column] for label in labels})
        table.insert(0, "site_no", site)
        table.reset_index().to_csv(out / name, sep="\t", index=False)


def write_analysis_frame(out: Path) -> None:
    """Sample statistics joined to station metadata and measured sieve percentages."""
    from psd_gig.paper_data import load_samples
    frame = load_samples(with_sieve_columns=True)
    frame.to_csv(out / "analysis_frame.csv", index=False)
    return frame


def matrix(frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    return pd.DataFrame({label: frame[column] for label, frame in frames.items()})


def upper_outliers(values: pd.Series) -> int:
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    return int((values > q3 + 1.5 * (q3 - q1)).sum())


def best_counts(bic: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    minimum = bic.min(axis=1)
    rows = {"Smallest BIC": (bic.eq(minimum, axis=0)).sum()}
    for tolerance in TOLERANCES:
        rows[f"Difference BIC < {tolerance}"] = (
            bic.le(minimum + tolerance, axis=0)).sum()
    return pd.DataFrame(rows).reindex(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", type=Path, default=ROOT / "data" / "fitted_functions")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "tables")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = load_metrics(args.fits, keep_site=True)
    labels = [spec.label for spec in FUNCTION_SPECS]
    n_params = {spec.label: spec.n_params for spec in FUNCTION_SPECS}
    bic = matrix(frames, "BIC")
    rmse = matrix(frames, "RMSE")
    r2 = matrix(frames, "R2")
    print(f"{len(bic):,} samples x {len(labels)} functions")

    # ---- Table S2 -------------------------------------------------------
    s2 = pd.DataFrame({
        "no": [spec.number for spec in FUNCTION_SPECS],
        "function": labels,
        "n_parameters": [n_params[label] for label in labels],
        "median_BIC": [bic[label].median() for label in labels],
        "n_BIC_upper_outliers": [upper_outliers(bic[label]) for label in labels],
        "median_RMSE": [rmse[label].median() for label in labels],
        "median_R2": [r2[label].median() for label in labels],
    })
    s2.to_csv(args.out / "table_S2_median_metrics.csv", index=False)

    # ---- Tables S3 and S4 -----------------------------------------------
    s3 = best_counts(bic, labels).reset_index().rename(columns={"index": "function"})
    s3.insert(0, "no", [spec.number for spec in FUNCTION_SPECS])
    s3.to_csv(args.out / "table_S3_best_counts.csv", index=False)

    s4_parts = []
    for size in (2, 3):
        family = [label for label in labels if n_params[label] == size]
        part = best_counts(bic[family], family).reset_index().rename(columns={"index": "function"})
        part.insert(0, "n_parameters", size)
        s4_parts.append(part)
    pd.concat(s4_parts).to_csv(args.out / "table_S4_by_n_parameters.csv", index=False)

    # ---- Figure 4 source data -------------------------------------------
    delta = bic.sub(bic.min(axis=1), axis=0)
    fig4 = pd.DataFrame({
        "function": labels,
        "n_parameters": [n_params[label] for label in labels],
        "median_delta_BIC": [delta[label].median() for label in labels],
        "q25_delta_BIC": [delta[label].quantile(0.25) for label in labels],
        "q75_delta_BIC": [delta[label].quantile(0.75) for label in labels],
        "p10_delta_BIC": [delta[label].quantile(0.10) for label in labels],
        "p90_delta_BIC": [delta[label].quantile(0.90) for label in labels],
        "lowest_BIC_share_percent": [
            float((bic.idxmin(axis=1) == label).mean() * 100) for label in labels],
    }).sort_values("median_delta_BIC")
    fig4.to_csv(args.out / "figure4_delta_bic_summary.csv", index=False)

    write_matrices(frames, labels, args.out)
    analysis = write_analysis_frame(args.out)
    print(f"\nintermediates: bic_matrix.tsv, rmse_matrix.tsv, "
          f"analysis_frame.csv ({len(analysis):,} rows x {len(analysis.columns)} cols)")

    top = s2.nsmallest(6, "median_BIC")
    print("\nTable S2 -- six best-ranked functions by median BIC")
    print(f"  {'function':11s} {'median BIC':>11s} {'median RMSE':>12s} {'median R2':>10s} "
          f"{'lowest-BIC share':>17s}")
    share = dict(zip(fig4["function"], fig4["lowest_BIC_share_percent"]))
    for _, row in top.iterrows():
        print(f"  {row['function']:11s} {row['median_BIC']:11.3f} {row['median_RMSE']:12.3f} "
              f"{row['median_R2']:10.3f} {share[row['function']]:16.2f}%")
    print(f"\n  wrote table_S2, table_S3, table_S4, figure4_delta_bic_summary "
          f"-> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
