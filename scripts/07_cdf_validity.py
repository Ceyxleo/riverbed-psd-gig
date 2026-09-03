#!/usr/bin/env python3
"""Are the fitted candidates actually cumulative distributions?

Each function is fitted by unconstrained non-linear least squares to the measured
cumulative curve. Least squares does not enforce that the result is monotone or
that it stays within 0-100%, so a fitted "distribution" can fail to be one. This
checks each archived fit inside the sieve range that sample actually measured --
no extrapolation -- and counts three failure modes:

    non-monotone      the curve decreases by more than 0.5 percentage points
    above 100%        the curve exceeds 100.5%
    below 0%          the curve drops below -0.5%

Every function the paper's conclusions rest on comes through clean. Several of
the also-ran candidates do not, which is worth stating alongside "we evaluated
25 candidate distributions".

    python scripts/07_cdf_validity.py --samples 500
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

from psd_gig.fit_specs import FUNCTION_SPECS  # noqa: E402

MONOTONE_TOLERANCE = 0.5
BOUND_TOLERANCE = 0.5


def measured_ranges(data_dir: Path) -> pd.DataFrame:
    samples = pd.read_csv(data_dir / "usgs_psd_samples_qc.csv", low_memory=False)
    sizes = pd.read_csv(data_dir / "reference" / "sieve_sizes_all43.csv")
    lookup = {column: float(sizes[column].iloc[0]) for column in sizes.columns}
    columns = [c for c in samples.columns if c in lookup]
    values = samples[columns].to_numpy(float)
    grid = np.array([lookup[c] for c in columns])
    order = np.argsort(grid)
    values, grid = values[:, order], grid[order]
    present = np.isfinite(values)
    first = grid[present.argmax(axis=1)]
    last = grid[present.shape[1] - 1 - present[:, ::-1].argmax(axis=1)]
    return pd.DataFrame({"sample_ID": samples["sample_ID"].astype(int),
                         "d_min": first, "d_max": last}).set_index("sample_ID")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--fits", type=Path, default=ROOT / "data" / "fitted_functions")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "tables" / "cdf_validity.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    ranges = measured_ranges(args.data)

    rows = []
    for spec in FUNCTION_SPECS:
        fit = pd.read_csv(args.fits / f"fits_F{spec.number:02d}_{spec.label}.csv")
        fit = fit.dropna(subset=["fitted_A"])
        fit["sample_ID"] = fit["sample_ID"].astype(int)
        subset = fit.sample(n=min(args.samples, len(fit)), random_state=0)
        columns = ["fitted_A", "fitted_B", "fitted_C"][: spec.n_params]

        checked = non_monotone = above = below = 0
        worst_drop = 0.0
        worst_max = 100.0
        for _, row in subset.iterrows():
            if row["sample_ID"] not in ranges.index:
                continue
            bounds = ranges.loc[row["sample_ID"]]
            parameters = [row[c] for c in columns]
            if not np.isfinite([bounds.d_min, bounds.d_max]).all() or not all(
                    np.isfinite(parameters)):
                continue
            grid = np.logspace(np.log2(bounds.d_min), np.log2(bounds.d_max), 80, base=2)
            x = np.log2(grid) if spec.base_x_transform == "log2" else grid
            with np.errstate(all="ignore"):
                y = np.asarray(spec.func(x, *parameters), dtype=float)
            y = y[np.isfinite(y)]
            if len(y) < 20:
                continue
            checked += 1
            step = float(np.diff(y).min())
            if step < -MONOTONE_TOLERANCE:
                non_monotone += 1
                worst_drop = min(worst_drop, step)
            if y.max() > 100 + BOUND_TOLERANCE:
                above += 1
                worst_max = max(worst_max, float(y.max()))
            if y.min() < -BOUND_TOLERANCE:
                below += 1
        rows.append({
            "number": spec.number, "function": spec.label,
            "n_parameters": spec.n_params, "grid_searched": spec.grid is not None,
            "n_checked": checked,
            "pct_non_monotone": non_monotone / checked * 100 if checked else np.nan,
            "pct_above_100": above / checked * 100 if checked else np.nan,
            "pct_below_0": below / checked * 100 if checked else np.nan,
            "worst_decrease_pp": worst_drop, "worst_maximum_percent": worst_max,
            "valid_distribution": checked > 0 and non_monotone == above == below == 0,
        })

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"checked up to {args.samples} archived fits per function, "
          f"inside each sample's measured sieve range\n")
    print(f"{'F':>3} {'function':11s} {'p':>1} {'grid':>5s} {'non-mono':>9s} "
          f"{'>100%':>7s} {'<0%':>6s} {'worst max':>10s}  verdict")
    for _, row in table.iterrows():
        verdict = "valid CDF" if row["valid_distribution"] else "NOT a valid CDF"
        print(f"{row['number']:3d} {row['function']:11s} {row['n_parameters']:1d} "
              f"{'yes' if row['grid_searched'] else '-':>5s} "
              f"{row['pct_non_monotone']:8.1f}% {row['pct_above_100']:6.1f}% "
              f"{row['pct_below_0']:5.1f}% {row['worst_maximum_percent']:9.1f}%  {verdict}")
    valid = table.loc[table["valid_distribution"], "function"].tolist()
    print(f"\nvalid throughout: {', '.join(valid)}")
    print(f"  -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
