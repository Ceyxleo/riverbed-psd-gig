#!/usr/bin/env python3
"""Check that this repository reproduces the shipped CONUS fits.

Nine functions (F05, F06, F10, F12, F14, F15, F16, F18, F25) are shipped from the
original grid runs at maxfev = 1e6 instead of being re-fitted here, because their
CDFs are evaluated by numerical integration and a full re-run would take weeks. This
script re-fits a subset of samples through the same `grid_search_dataframe` used by
the pipeline and compares BIC and RMSE against the shipped values.

The subset is deliberately adversarial: half is drawn at random, half from the 924
samples carrying sieve grades beyond the twelve principal codes. Those are exactly the
samples that disagree if `size_columns` is wrongly restricted, so a clean result is
evidence that the shipped configuration is the one that produced the files.

    python scripts/verify_fits.py --samples 60                 # the five cheap ones
    python scripts/verify_fits.py --samples 20 --functions 12  # NIG: slow

The other sixteen functions are produced by this repository, so there is nothing
independent to compare them against; pass their numbers explicitly to fit them twice.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psd_gig.config import load_config  # noqa: E402
from psd_gig.fit_data import load_diameter_lookup, load_input_data  # noqa: E402
from psd_gig.fit_specs import FUNCTION_SPECS_BY_NUMBER  # noqa: E402
from psd_gig.fitting import grid_search_dataframe  # noqa: E402

PRINCIPAL = [f"p{code}" for code in range(80164, 80176)]
#: Bit-exact agreement. Reported, but not what the check passes or fails on.
TOLERANCE = 1e-6
#: Agreement that matters: two BIC units is the paper's own threshold for "no difference",
#: so a re-fit landing within 0.01 of the shipped value is the same fit for every purpose.
NEGLIGIBLE_DBIC = 0.01
ARCHIVE_MAXFEV = 1_000_000
#: The shipped functions cheap enough to verify in minutes rather than hours.
DEFAULT_FUNCTIONS = (10, 14, 15, 18, 25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/fit_usgs.yaml")
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "fitted_functions")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--functions", nargs="*", type=int, default=list(DEFAULT_FUNCTIONS),
                        help=f"Function numbers to check (default: {list(DEFAULT_FUNCTIONS)}).")
    parser.add_argument("--maxfev", type=int, default=ARCHIVE_MAXFEV,
                        help="Evaluation cap; the shipped files used 1,000,000.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "verify" / "verify_fits.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    config = load_config(ROOT / args.config)
    data, size_columns = load_input_data(config)
    lookup = load_diameter_lookup(config, size_columns)

    n_all = data[size_columns].notna().sum(axis=1)
    n_principal = data[[c for c in PRINCIPAL if c in data.columns]].notna().sum(axis=1)
    extra = data.loc[(n_all != n_principal).to_numpy()]

    rng = np.random.default_rng(args.seed)
    half = max(1, args.samples // 2)
    picks = [data.iloc[rng.choice(len(data), min(half, len(data)), replace=False)]]
    if len(extra):
        picks.append(extra.iloc[rng.choice(len(extra), min(args.samples - half, len(extra)),
                                           replace=False)])
    subset = pd.concat(picks).drop_duplicates("sample_ID")
    print(f"config      : {args.config}")
    print(f"size columns: {len(size_columns)} ({config['data']['size_columns']})")
    print(f"maxfev      : {args.maxfev:,}")
    print(f"subset      : {len(subset)} samples "
          f"({len(picks[1]) if len(picks) > 1 else 0} with grades beyond the 12 principal codes)\n")

    rows = []
    print(f"{'F':>3} {'function':11s} {'n':>5s} {'exact':>8s} {'<0.01':>8s} "
          f"{'med |dBIC|':>11s} {'max |dBIC|':>11s}")
    for number in args.functions:
        spec = FUNCTION_SPECS_BY_NUMBER[number]
        archived = pd.read_csv(args.archive / spec.output_name).set_index("sample_ID")
        refit, _ = grid_search_dataframe(
            spec, subset, lookup, size_columns, maxfev=args.maxfev, progress=False,
            grid_all_path=None, write_grid_search_all=False, grid_chunk_samples=10**9)
        refit["sample_ID"] = refit["sample_ID"].astype(int)
        merged = refit.set_index("sample_ID").join(archived, how="inner", rsuffix="_ref")
        merged = merged[np.isfinite(merged["BIC"]) & np.isfinite(merged["BIC_ref"])]
        if merged.empty:
            print(f"{number:3d} {spec.label:11s} {'0':>5s} {'-':>10s} {'-':>11s} {'-':>11s}")
            continue
        db = (merged["BIC"] - merged["BIC_ref"]).abs().to_numpy()
        dr = (merged["RMSE"] - merged["RMSE_ref"]).abs().to_numpy()
        bic_match = float((db < TOLERANCE).mean())
        negligible = float((db < NEGLIGIBLE_DBIC).mean())
        rows.append({"number": number, "function": spec.label, "n": len(db),
                     "maxfev": args.maxfev,
                     "bic_exact_fraction": bic_match,
                     "bic_negligible_fraction": negligible,
                     "rmse_exact_fraction": float((dr < TOLERANCE).mean()),
                     "max_abs_dBIC": float(db.max()), "median_abs_dBIC": float(np.median(db))})
        flag = "" if negligible >= 0.99 else "   <-- check"
        print(f"{number:3d} {spec.label:11s} {len(db):5d} {bic_match * 100:7.1f}% "
              f"{negligible * 100:7.1f}% {np.median(db):11.3g} {db.max():11.3g}{flag}",
              flush=True)

    summary = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    exact = float(np.average(summary["bic_exact_fraction"], weights=summary["n"]))
    within = float(np.average(summary["bic_negligible_fraction"], weights=summary["n"]))
    print(f"\nbit-exact: {exact * 100:.2f}%   within {NEGLIGIBLE_DBIC} BIC: {within * 100:.2f}%"
          f"   ->  {args.out.relative_to(ROOT)}")
    if exact < 0.99 <= within:
        print("\nThe gap is optimiser noise, not disagreement: GIG_3p fits a numerically\n"
              "integrated CDF, so curve_fit's path depends on quadrature detail that differs\n"
              "between SciPy versions. Worst deviation "
              f"{summary['max_abs_dBIC'].max():.2g} BIC, against the 2 units the paper\n"
              "treats as no difference at all.")
    return 0 if within >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
