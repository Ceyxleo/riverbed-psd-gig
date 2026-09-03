#!/usr/bin/env python3
"""Check that `configs/fit_usgs.yaml` reproduces the archived fitted parameters.

The archived USGS fits took several days of wall clock to produce and are shipped
in the data deposit rather than regenerated here. This script re-fits a subset of
samples with the configuration in this repository and compares BIC and RMSE
against the archived values, function by function.

The subset is deliberately adversarial: half is drawn at random, half is drawn
from the 924 samples that carry sieve grades beyond the twelve principal codes.
Those are exactly the samples that disagree if `size_columns` is wrongly
restricted to the twelve principal codes, so a clean result here is evidence
that the shipped configuration is the one that produced the archive.

    python scripts/verify_fits.py --samples 200
    python scripts/verify_fits.py --samples 60 --functions 14 10 15
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
from psd_gig.fit_specs import FUNCTION_SPECS  # noqa: E402
from psd_gig.fitting import fit_one_sample  # noqa: E402

PRINCIPAL = [f"p{code}" for code in range(80164, 80176)]
TOLERANCE = 1e-6


def sample_xy(row, lookup, size_columns, transform):
    y = pd.to_numeric(row[size_columns], errors="coerce").dropna()
    d = lookup.loc[y.index, "d"].to_numpy(dtype=float)
    return (np.log2(d) if transform == "log2" else d), y.to_numpy(dtype=float)


def fit_sample(spec, x, y, maxfev):
    if spec.grid is None:
        return fit_one_sample(spec, x, y, p0=spec.base_initial_guess,
                              retry_p0=spec.base_retry_initial_guess, maxfev=maxfev)
    best = None
    for p0 in spec.grid.initial_values():
        row = fit_one_sample(spec, x, y, p0=p0, retry_p0=None, maxfev=maxfev)
        bic = row.get("BIC", np.nan)
        if np.isfinite(bic) and (best is None or bic < best["BIC"]):
            best = row
    return best if best is not None else {"BIC": np.nan, "RMSE": np.nan}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/fit_usgs.yaml")
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "fitted_functions")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--functions", nargs="*", type=int,
                        help="Function numbers to check (default: all 25).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "verify" / "verify_fits.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    config = load_config(ROOT / args.config)
    data, size_columns = load_input_data(config)
    lookup = load_diameter_lookup(config, size_columns)
    maxfev = int(config["runtime"]["maxfev"])

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
    print(f"maxfev      : {maxfev:,}")
    print(f"subset      : {len(subset)} samples "
          f"({len(picks[1]) if len(picks) > 1 else 0} with grades beyond the 12 principal codes)\n")

    specs = [s for s in FUNCTION_SPECS
             if args.functions is None or s.number in set(args.functions)]
    rows = []
    print(f"{'F':>3} {'function':11s} {'n':>5s} {'BIC match':>10s} {'RMSE match':>11s} "
          f"{'max |dBIC|':>11s}")
    for spec in specs:
        archived = pd.read_csv(args.archive / f"fits_F{spec.number:02d}_{spec.label}.csv")
        archived = archived.set_index("sample_ID")
        deltas_bic, deltas_rmse = [], []
        for _, row in subset.iterrows():
            key = int(row["sample_ID"])
            if key not in archived.index:
                continue
            x, y = sample_xy(row, lookup, size_columns, spec.base_x_transform)
            fit = fit_sample(spec, x, y, maxfev)
            ref = archived.loc[key]
            if np.isfinite(fit.get("BIC", np.nan)) and np.isfinite(ref["BIC"]):
                deltas_bic.append(abs(fit["BIC"] - ref["BIC"]))
                deltas_rmse.append(abs(fit["RMSE"] - ref["RMSE"]))
        if not deltas_bic:
            print(f"{spec.number:3d} {spec.label:11s} {'0':>5s} {'-':>10s} {'-':>11s} {'-':>11s}")
            continue
        db = np.asarray(deltas_bic)
        dr = np.asarray(deltas_rmse)
        bic_match = float((db < TOLERANCE).mean())
        rmse_match = float((dr < TOLERANCE).mean())
        rows.append({"number": spec.number, "function": spec.label, "n": len(db),
                     "bic_match_fraction": bic_match, "rmse_match_fraction": rmse_match,
                     "max_abs_dBIC": float(db.max()), "median_abs_dBIC": float(np.median(db))})
        flag = "" if bic_match >= 0.99 else "   <-- check"
        print(f"{spec.number:3d} {spec.label:11s} {len(db):5d} {bic_match * 100:9.1f}% "
              f"{rmse_match * 100:10.1f}% {db.max():11.3g}{flag}")

    summary = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    overall = float(np.average(summary["bic_match_fraction"], weights=summary["n"]))
    print(f"\noverall BIC agreement: {overall * 100:.2f}%   ->  {args.out.relative_to(ROOT)}")
    return 0 if overall >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
