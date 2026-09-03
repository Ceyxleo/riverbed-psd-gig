#!/usr/bin/env python3
"""Invert each fitted CDF for representative grain-size percentiles.

For every sample and every one of GIG_2p, Lognormal and Weibull, solve
``F(D) = q`` for D at each requested percentile. The *reference* value for a
sample is the mean over the functions in its best-fit set, which is the benchmark
that Fig. 5 compares the three individual functions against.

Percentiles 25 and 75 are needed for the Fredle Index, 16/50/84 for Fig. 5, and
5/95 for the tail-coverage diagnostics.

    python scripts/02_compute_dvalues.py
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import brentq
from scipy.stats import geninvgauss

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = ("GIG_2p", "Lognormal", "Weibull")
FIT_FILES = {
    "GIG_2p": "fits_F14_GIG_2p.csv",
    "Lognormal": "fits_F10_Lognormal.csv",
    "Weibull": "fits_F15_Weibull.csv",
}
DEFAULT_TARGETS = (5, 16, 25, 50, 75, 84, 95)


def finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def lognormal_values(fit: pd.DataFrame, percent: float):
    sigma = fit["fitted_A"].to_numpy(float)
    mu = fit["fitted_B"].to_numpy(float)
    with np.errstate(all="ignore"):
        d = np.exp(mu + sigma * special.ndtri(percent / 100))
        y = 0.5 * (1 + special.erf((np.log(d) - mu) / (sigma * np.sqrt(2)))) * 100
    return finite(d), finite(y)


def weibull_values(fit: pd.DataFrame, percent: float):
    lam = fit["fitted_A"].to_numpy(float)
    k = fit["fitted_B"].to_numpy(float)
    with np.errstate(all="ignore"):
        d = lam * np.power(-np.log1p(-percent / 100), 1 / k)
        y = (1 - np.exp(-np.power(d / lam, k))) * 100
    bad = (lam <= 0) | (k <= 0) | ~np.isfinite(d) | (d <= 0) | ~np.isfinite(y)
    return finite(np.where(bad, np.nan, d)), finite(np.where(bad, np.nan, y))


def gig_scalar_ppf(percent: float, eta: float, beta: float) -> float:
    """Robust inverse for one sample: analytic ppf, then bracketed root, then scan."""
    p = percent / 100
    try:
        value = float(geninvgauss.ppf(p, eta, beta))
        if math.isfinite(value) and value > 0:
            return value
    except Exception:
        pass

    upper = 1.0
    for _ in range(120):
        try:
            if float(geninvgauss.cdf(upper, eta, beta)) >= p:
                break
        except Exception:
            pass
        upper *= 2
    else:
        return np.nan
    try:
        return float(brentq(lambda x: float(geninvgauss.cdf(x, eta, beta)) - p,
                            0, upper, xtol=1e-10, rtol=1e-10, maxiter=1000))
    except Exception:
        return np.nan


def gig_values(fit: pd.DataFrame, percent: float):
    eta = fit["fitted_A"].to_numpy(float)
    beta = fit["fitted_B"].to_numpy(float)
    with np.errstate(all="ignore"):
        d = np.asarray(geninvgauss.ppf(percent / 100, eta, beta), dtype=float)
    bad = ~np.isfinite(d) | (d <= 0)
    for i in np.where(bad)[0]:
        d[i] = gig_scalar_ppf(percent, eta[i], beta[i])
    with np.errstate(all="ignore"):
        y = np.asarray(geninvgauss.cdf(d, eta, beta), dtype=float) * 100

    # Retry anything that did not land on the requested percentile.
    off = np.isfinite(d) & np.isfinite(y) & (np.abs(y - percent) > 0.01)
    for i in np.where(off | ~np.isfinite(y))[0]:
        value = gig_scalar_ppf(percent, eta[i], beta[i])
        if np.isfinite(value):
            d[i] = value
            y[i] = float(geninvgauss.cdf(value, eta[i], beta[i])) * 100
    bad = ~np.isfinite(d) | (d <= 0) | ~np.isfinite(y)
    return finite(np.where(bad, np.nan, d)), finite(np.where(bad, np.nan, y))


SOLVERS = {"GIG_2p": gig_values, "Lognormal": lognormal_values, "Weibull": weibull_values}


def reference_column(frame: pd.DataFrame, label: str) -> np.ndarray:
    matrix = frame[[f"{label}_{f}" for f in FUNCTIONS]].to_numpy(float)
    index = {f: i for i, f in enumerate(FUNCTIONS)}
    out = np.full(len(frame), np.nan)
    for i, best in enumerate(frame["best_function(s)"]):
        picks = [index[f.strip()] for f in str(best).split("&") if f.strip() in index]
        if picks:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[i] = np.nanmean(matrix[i, picks])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, default=ROOT / "data" / "usgs_sample_statistics.csv")
    parser.add_argument("--fits", type=Path, default=ROOT / "data" / "fitted_functions")
    parser.add_argument("--targets", nargs="+", type=float, default=list(DEFAULT_TARGETS))
    parser.add_argument("--archived", type=Path,
                        default=ROOT / "data" / "dvalues_archived.csv",
                        help="Archived percentile table to compare against, if present.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "dvalues")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.out.mkdir(parents=True, exist_ok=True)

    stats = pd.read_csv(args.stats, dtype={"site_no": str})
    stats["sample_ID"] = stats["sample_ID"].astype(int)
    table = stats[["sample_ID", "site_no", "best_function(s)", "num_functions", "skew_FW",
                   "stream_type", "sample_ST_type"]].copy()

    fits = {}
    for name, filename in FIT_FILES.items():
        fit = pd.read_csv(args.fits / filename, dtype={"site_no": str})
        fit["sample_ID"] = fit["sample_ID"].astype(int)
        fits[name] = fit.drop_duplicates("sample_ID").set_index("sample_ID").reindex(
            table["sample_ID"].to_numpy())

    qc = []
    for target in args.targets:
        label = f"D{int(target):02d}" if float(target).is_integer() else f"D{target}".replace(".", "p")
        for name in FUNCTIONS:
            d, y = SOLVERS[name](fits[name], float(target))
            table[f"{label}_{name}"] = d
            qc.append({"target": label, "function": name, "n": len(d),
                       "n_missing": int(np.isnan(d).sum()),
                       "min_d_mm": np.nanmin(d), "max_d_mm": np.nanmax(d),
                       "min_solved_percent": np.nanmin(y), "max_solved_percent": np.nanmax(y)})
        table[f"{label}_reference"] = reference_column(table, label)
        print(f"  {label}: solved  " + "  ".join(
            f"{n}={int(table[f'{label}_{n}'].notna().sum()):,}" for n in FUNCTIONS)
            + f"  reference={int(table[f'{label}_reference'].notna().sum()):,}")

    ordered = ["sample_ID", "site_no", "best_function(s)", "num_functions", "skew_FW",
               "stream_type", "sample_ST_type"]
    for target in args.targets:
        label = f"D{int(target):02d}" if float(target).is_integer() else f"D{target}".replace(".", "p")
        ordered += [f"{label}_reference"] + [f"{label}_{n}" for n in FUNCTIONS]
    out_path = args.out / "dvalues_recomputed.csv"
    table[ordered].to_csv(out_path, index=False)
    pd.DataFrame(qc).to_csv(args.out / "dvalues_qc_summary.csv", index=False)
    print(f"\n{len(table):,} samples -> {out_path.relative_to(ROOT)}")

    if args.archived.exists():
        compare(table, pd.read_csv(args.archived, dtype={"site_no": str}), args.targets, args.out)
    return 0


def compare(recomputed: pd.DataFrame, archived: pd.DataFrame, targets, out: Path) -> None:
    """Quantify the analytic inverse against the archived grid-scan percentiles."""
    archived["sample_ID"] = archived["sample_ID"].astype(int)
    rows = []
    for target in targets:
        label = f"D{int(target):02d}" if float(target).is_integer() else f"D{target}".replace(".", "p")
        for name in list(FUNCTIONS) + ["reference"]:
            column = f"{label}_{name}"
            if column not in archived.columns:
                continue
            merged = recomputed[["sample_ID", column]].merge(
                archived[["sample_ID", column]], on="sample_ID", suffixes=("_new", "_old"))
            a = merged[f"{column}_new"]
            b = merged[f"{column}_old"]
            ok = a.notna() & b.notna() & (b.abs() > 0)
            relative = ((a[ok] - b[ok]).abs() / b[ok].abs())
            rows.append({"target": label, "function": name, "n": int(ok.sum()),
                         "median_rel_diff": float(relative.median()),
                         "p99_rel_diff": float(relative.quantile(0.99)),
                         "max_rel_diff": float(relative.max()),
                         "n_rel_diff_gt_1pct": int((relative > 0.01).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "dvalues_archived_vs_recomputed.csv", index=False)
    worst = summary.loc[summary["max_rel_diff"].idxmax()]
    print(f"vs archived: median relative difference "
          f"{summary['median_rel_diff'].median():.2g}; "
          f"largest single disagreement {worst['max_rel_diff']:.3g} "
          f"({worst['target']} {worst['function']}); "
          f"{int(summary['n_rel_diff_gt_1pct'].sum())} values differ by more than 1%")
    print("  -> outputs/dvalues/dvalues_archived_vs_recomputed.csv  (see docs/caveats.md)")


if __name__ == "__main__":
    raise SystemExit(main())
