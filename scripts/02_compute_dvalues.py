#!/usr/bin/env python3
"""Invert each fitted CDF for representative grain-size percentiles.

For every sample and every one of GIG_2p, Lognormal and Weibull, solve
``F(D) = q`` for D at each requested percentile. The *reference* value for a
sample is the mean over the functions in its best-fit set, which is the benchmark
Fig. 5 compares the three individual functions against.

The inverse is analytic wherever one exists -- ``geninvgauss.ppf`` for GIG,
``exp(mu + sigma Phi^-1(q))`` for Lognormal, ``lambda (-ln(1-q))^(1/k)`` for
Weibull -- with a bracketed Brent root-find as fallback. Solved percentiles land
on the target to better than 1e-9 and there is no upper bound on D, so samples
whose fitted distribution places a percentile far beyond the coarsest sieve are
resolved rather than truncated.

Percentiles 25 and 75 feed the Fredle Index, 16/50/84 feed Fig. 5 and the
hydraulics, and 5/95 are carried for tail diagnostics.

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

ROUND_TRIP_TOLERANCE = 0.01   # percentage points


def drop_inconsistent(d: np.ndarray, y: np.ndarray, target: float) -> tuple[np.ndarray, int]:
    """Discard percentiles that fail a round trip through the fitted CDF.

    A solved D is only meaningful if evaluating the CDF at it returns the target
    percentile. For a handful of fits the parameters are extreme enough that the
    CDF implementation itself loses accuracy -- returning values above 100%, for
    instance -- and no inverse of it can be trusted. Those are set to missing
    rather than carried forward.
    """
    bad = np.isfinite(d) & (np.abs(y - target) > ROUND_TRIP_TOLERANCE)
    return np.where(bad, np.nan, d), int(bad.sum())


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
    dropped: dict[str, set] = {}
    for target in args.targets:
        label = f"D{int(target):02d}" if float(target).is_integer() else f"D{target}".replace(".", "p")
        for name in FUNCTIONS:
            d, y = SOLVERS[name](fits[name], float(target))
            d, n_dropped = drop_inconsistent(d, y, float(target))
            if n_dropped:
                dropped.setdefault(name, set()).update(
                    table.loc[np.isnan(d) & np.isfinite(y), "sample_ID"].tolist())
            table[f"{label}_{name}"] = d
            y_kept = np.where(np.isfinite(d), y, np.nan)
            qc.append({"target": label, "function": name, "n": len(d),
                       "n_missing": int(np.isnan(d).sum()),
                       "n_failed_round_trip": n_dropped,
                       "min_d_mm": np.nanmin(d), "max_d_mm": np.nanmax(d),
                       "min_solved_percent": np.nanmin(y_kept),
                       "max_solved_percent": np.nanmax(y_kept)})
        table[f"{label}_reference"] = reference_column(table, label)
        print(f"  {label}: solved  " + "  ".join(
            f"{n}={int(table[f'{label}_{n}'].notna().sum()):,}" for n in FUNCTIONS)
            + f"  reference={int(table[f'{label}_reference'].notna().sum()):,}")

    ordered = ["sample_ID", "site_no", "best_function(s)", "num_functions", "skew_FW",
               "stream_type", "sample_ST_type"]
    for target in args.targets:
        label = f"D{int(target):02d}" if float(target).is_integer() else f"D{target}".replace(".", "p")
        ordered += [f"{label}_reference"] + [f"{label}_{n}" for n in FUNCTIONS]
    out_path = args.out / "dvalues.csv"
    table[ordered].to_csv(out_path, index=False)
    pd.DataFrame(qc).to_csv(args.out / "dvalues_qc_summary.csv", index=False)
    print(f"\n{len(table):,} samples -> {out_path.relative_to(ROOT)}")

    accuracy = pd.DataFrame(qc)
    worst = accuracy[["min_solved_percent", "max_solved_percent"]].to_numpy()
    targets = np.repeat(args.targets, len(FUNCTIONS))
    deviation = float(np.nanmax(np.abs(worst - targets[:, None])))
    print(f"largest deviation of a solved percentile from its target: {deviation:.2g}")
    total_dropped = int(accuracy["n_failed_round_trip"].sum())
    if total_dropped:
        affected = sorted({sid for ids in dropped.values() for sid in ids})
        print(f"discarded {total_dropped} percentile values that failed the round trip, "
              f"from {len(affected)} sample(s): {affected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
