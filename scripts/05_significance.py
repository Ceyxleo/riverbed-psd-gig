#!/usr/bin/env python3
"""Supplementary Table S5: paired tests on the error reduction achieved by GIG.

For each water-security variable, the squared relative error of a sample under
function f is

    SRE_f = ((X_reference - X_f) / X_reference)^2

and the table reports 100 * mean(SRE) with its standard deviation, then a paired
Wilcoxon signed-rank test on (SRE_other - SRE_GIG) with the one-sided alternative
that the difference is greater than zero, i.e. that GIG has the smaller error.

Three things this fixes relative to the submitted Table S5:

* the "SRE, %" heading was ambiguous. The quantity is 100 x the mean of a squared
  *fractional* relative error, which is identically (RMSRE in %)^2 / 100. The
  column is labelled accordingly here.
* the test statistic column was headed "t-statistics", which is wrong for a
  signed-rank test. The Wilcoxon W is reported.
* no statistic was actually given, only p-values. With n between 12,930 and
  19,763 every p-value is far below any threshold, so the effect size (matched-
  pairs rank-biserial correlation) and the win rate are reported alongside; they
  are what actually distinguish the comparisons.

    python scripts/05_significance.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

VARIABLES = [
    ("Bedload transport", "bedload_transport_", "_m2_s", "bedload_transport"),
    ("Critical shear stress", "critical_shear_stress_", "_Pa", "critical_shear_stress"),
    ("100-year flood stage", "flood_stage_", "_m", "flood_stage_100yr"),
    ("Fredle Index", "fredle_index_", "", "fredle_index"),
]
FUNCTIONS = ("GIG_2p", "Lognormal", "Weibull")


def subset(frame: pd.DataFrame, prefix: str, suffix: str, name: str):
    reference = f"{prefix}reference{suffix}"
    columns = [reference] + [f"{prefix}{f}{suffix}" for f in FUNCTIONS]
    rows = frame.loc[frame[columns].notna().all(axis=1)]
    if name == "bedload_transport":
        rows = rows.loc[rows[reference] != rows[reference].min()]
    return rows.loc[rows[reference] != 0], reference


def squared_relative_error(reference: pd.Series, prediction: pd.Series) -> pd.Series:
    return (((reference - prediction) / reference) ** 2) * 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path,
                        default=ROOT / "outputs" / "water" / "sample_water_metrics.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "tables")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.metrics, dtype={"site_no": str}, low_memory=False)

    rows = []
    for label, prefix, suffix, name in VARIABLES:
        data, reference = subset(frame, prefix, suffix, name)
        errors = {f: squared_relative_error(data[reference], data[f"{prefix}{f}{suffix}"])
                  for f in FUNCTIONS}
        row = {"variable": label, "n_samples": len(data),
               "n_sites": data["site_no"].nunique()}
        for f in FUNCTIONS:
            row[f"meanSRE_{f}"] = float(errors[f].mean())
            row[f"sdSRE_{f}"] = float(errors[f].std(ddof=1))
            row[f"RMSRE_{f}_percent"] = float(np.sqrt(errors[f].mean() / 100) * 100)
        rows.append(row)

        for other in ("Lognormal", "Weibull"):
            difference = (errors[other] - errors["GIG_2p"]).to_numpy()
            w, p = stats.wilcoxon(difference, zero_method="wilcox", alternative="greater")
            nonzero = int(np.sum(difference != 0))
            row[f"wilcoxon_W_vs_{other}"] = float(w)
            row[f"wilcoxon_p_one_sided_vs_{other}"] = float(p)
            row[f"n_nonzero_pairs_vs_{other}"] = nonzero
            row[f"gig_lower_error_share_vs_{other}_percent"] = float((difference > 0).mean() * 100)
            row[f"rank_biserial_vs_{other}"] = float(
                (difference > 0).mean() - (difference < 0).mean())

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "table_S5_significance.csv", index=False)

    print("Table S5 -- mean (SD) squared relative error, x10^-2, and paired Wilcoxon tests\n")
    print(f"{'variable':22s} {'n':>7s} " + "".join(f"{f:>18s}" for f in FUNCTIONS))
    for _, row in table.iterrows():
        print(f"{row['variable']:22s} {row['n_samples']:7,} " +
              "".join(f"{row[f'meanSRE_{f}']:11.3f} ({row[f'sdSRE_{f}']:5.2f})"
                      for f in FUNCTIONS))
    print(f"\n{'variable':22s} {'contrast':12s} {'W':>15s} {'p':>11s} "
          f"{'GIG better':>11s} {'rank-biserial':>14s}")
    for _, row in table.iterrows():
        for other in ("Lognormal", "Weibull"):
            p = row[f"wilcoxon_p_one_sided_vs_{other}"]
            p_text = "< 1e-300" if p == 0 else f"{p:.2e}"
            print(f"{row['variable']:22s} {'vs ' + other:12s} "
                  f"{row[f'wilcoxon_W_vs_{other}']:15,.0f} {p_text:>11s} "
                  f"{row[f'gig_lower_error_share_vs_{other}_percent']:10.1f}% "
                  f"{row[f'rank_biserial_vs_{other}']:+14.3f}")
    print(f"\n  -> {(args.out / 'table_S5_significance.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
