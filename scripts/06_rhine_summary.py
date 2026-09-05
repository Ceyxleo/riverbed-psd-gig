#!/usr/bin/env python3
"""Lower Rhine independent validation: the numbers quoted in the main text.

Reads the fits produced by `01_fit_functions.py --config configs/fit_rhine.yaml`
and reports, for the 67 Lower Rhine samples:

* median BIC and RMSE for every one of the 25 candidate functions;
* the best-fit set per sample among GIG_2p, Lognormal and Weibull, using the
  same rule as the U.S. analysis -- a function is in the set when its BIC is
  within 2 units of the sample minimum;
* Folk-Ward skewness per sample, computed here from the measured cumulative
  curve rather than read from a file, so the step is self-contained given the
  user-supplied input; and the best-fit composition within each skewness class.

Manuscript targets: median BIC 10.07 (GIG), 13.99 (Lognormal), 9.16 (Weibull);
GIG best-fit for 52% of samples; 48 fine-skewed and 18 near-symmetric samples.

    python scripts/06_rhine_summary.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psd_gig.config import load_config  # noqa: E402
from psd_gig.fit_data import load_diameter_lookup, load_input_data  # noqa: E402
from psd_gig.fit_specs import FUNCTION_SPECS  # noqa: E402

THREE = ("GIG_2p", "Lognormal", "Weibull")
TIE_TOLERANCE = 2.0
TARGETS = {"GIG_2p": 10.07, "Lognormal": 13.99, "Weibull": 9.16}


def d_value(sizes: np.ndarray, percents: np.ndarray, target: float) -> float:
    """Percentile from the measured cumulative curve, linear in log2(D).

    Mirrors the convention used for the U.S. samples: interpolate between the
    bracketing measured points, and where the target falls outside the measured
    range extrapolate against a log2 bound of -8 or +8.
    """
    log2_sizes = np.log2(sizes)
    below = np.where(percents <= target)[0]
    above = np.where(percents >= target)[0]

    percent_low = percents[below[-1]] if len(below) else 0.0
    percent_high = percents[above[0]] if len(above) else 100.0
    size_low = log2_sizes[below[-1]] if len(below) else -8.0
    size_high = log2_sizes[above[0]] if len(above) else 8.0

    if size_low >= size_high:
        return float(2 ** size_low)
    if percent_low == 0.0 and percent_high == 100.0:
        return np.nan
    slope = (percent_high - percent_low) / (size_high - size_low)
    return float(2 ** ((target - percent_low) / slope + size_low))


def folk_ward(sizes: np.ndarray, percents: np.ndarray) -> dict[str, float]:
    """Folk & Ward (1957) graphic statistics, in phi = -log2(D).

    The phi percentile subscripts are complementary to the millimetre ones, because
    phi runs the opposite way to D: phi16 is -log2(D84), not -log2(D16). Getting this
    wrong leaves the graphic mean and kurtosis untouched but negates the sorting and
    the skewness, so `stand_div_FW` coming out negative is the tell.
    """
    d = {p: d_value(sizes, percents, p) for p in (5, 16, 25, 50, 75, 84, 95)}
    phi = {p: -np.log2(d[100 - p]) for p in (5, 16, 25, 50, 75, 84, 95)}
    spread = phi[84] - phi[16]
    tail = phi[95] - phi[5]
    return {
        "D50_mm": d[50],
        "mean_size_FW": float(2 ** -((phi[16] + phi[50] + phi[84]) / 3)),
        "stand_div_FW": float(spread / 4 + tail / 6.6),
        "skew_FW": float((phi[84] + phi[16] - 2 * phi[50]) / (2 * spread)
                         + (phi[5] + phi[95] - 2 * phi[50]) / (2 * tail)),
        "kurtosis_FW": float(tail / (2.44 * (phi[75] - phi[25]))),
    }


def skew_class(value: float) -> str:
    if value < -0.1:
        return "coarse-skewed"
    if value > 0.1:
        return "fine-skewed"
    return "near-symmetric"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/fit_rhine.yaml")
    parser.add_argument("--fits", type=Path,
                        default=ROOT / "outputs" / "fit_rhine" / "fitted_functions")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "rhine")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / args.config)
    data, size_columns = load_input_data(config)
    lookup = load_diameter_lookup(config, size_columns)

    # ---- Folk-Ward descriptors from the measured curves -------------------
    rows = []
    for _, sample in data.iterrows():
        values = pd.to_numeric(sample[size_columns], errors="coerce").dropna()
        sizes = lookup.loc[values.index, "d"].to_numpy(float)
        order = np.argsort(sizes)
        stats = folk_ward(sizes[order], values.to_numpy(float)[order])
        rows.append({"sample_ID": sample["sample_ID"], "site_no": sample["site_no"],
                     "n_points": len(values), **stats})
    samples = pd.DataFrame(rows)
    samples["skew_class"] = samples["skew_FW"].map(skew_class)

    # ---- per-function metrics --------------------------------------------
    metrics = {}
    for spec in FUNCTION_SPECS:
        path = args.fits / spec.output_name
        if not path.exists():
            print(f"  missing fit output for {spec.id}")
            continue
        fit = pd.read_csv(path)
        fit["sample_ID"] = fit["sample_ID"].astype(str)
        metrics[spec.label] = fit.drop_duplicates("sample_ID").set_index("sample_ID")
    if len(metrics) < len(FUNCTION_SPECS):
        print(f"\nonly {len(metrics)} of {len(FUNCTION_SPECS)} functions available; "
              f"run `python scripts/01_fit_functions.py --config {args.config}` first")

    bic = pd.DataFrame({label: frame["BIC"] for label, frame in metrics.items()})
    rmse = pd.DataFrame({label: frame["RMSE"] for label, frame in metrics.items()})
    summary = pd.DataFrame({
        "function": list(metrics),
        "n_parameters": [s.n_params for s in FUNCTION_SPECS if s.label in metrics],
        "n_samples": [int(bic[label].notna().sum()) for label in metrics],
        "median_BIC": [bic[label].median() for label in metrics],
        "median_RMSE": [rmse[label].median() for label in metrics],
        "median_R2": [metrics[label]["R2"].median() for label in metrics],
    }).sort_values("median_BIC")
    summary.to_csv(args.out / "rhine_function_summary.csv", index=False)

    # ---- best-fit sets among the three headline functions -----------------
    available = [f for f in THREE if f in bic.columns]
    three = bic[available].dropna()
    minimum = three.min(axis=1)
    within = three.le(minimum + TIE_TOLERANCE, axis=0)
    best_sets = within.apply(lambda row: " & ".join(row.index[row]), axis=1)
    samples = samples.merge(
        pd.DataFrame({"sample_ID": three.index, "best_function(s)": best_sets.to_numpy(),
                      "num_functions": within.sum(axis=1).to_numpy()}),
        on="sample_ID", how="left")
    samples.to_csv(args.out / "rhine_sample_table.csv", index=False)

    counts = (samples.groupby(["skew_class", "best_function(s)"]).size()
              .rename("n_samples").reset_index())
    counts.to_csv(args.out / "rhine_best_fit_by_skew_class.csv", index=False)

    # ---- report -----------------------------------------------------------
    n = len(samples)
    print(f"\nLower Rhine: {n} samples, {samples['site_no'].nunique()} reaches, "
          f"{int(samples['n_points'].min())} sieve points each\n")
    print("skewness classes:")
    for name, count in samples["skew_class"].value_counts().items():
        print(f"  {name:16s} {count:3d}  ({count / n * 100:.0f}%)")

    print("\nbest-fit share among GIG / Lognormal / Weibull "
          f"(BIC within {TIE_TOLERANCE:g} of the sample minimum):")
    for function in available:
        share = samples["best_function(s)"].fillna("").str.contains(function).mean()
        print(f"  {function:11s} {share * 100:5.1f}%")

    print("\nmedian BIC (manuscript target in brackets):")
    for function in available:
        target = TARGETS.get(function)
        note = f"   [paper: {target}]" if target else ""
        print(f"  {function:11s} {bic[function].median():7.3f}{note}")

    print("\nfive lowest median BIC across all fitted functions:")
    for _, row in summary.head(5).iterrows():
        print(f"  {row['function']:11s} p={row['n_parameters']}  "
              f"median BIC {row['median_BIC']:7.3f}  median RMSE {row['median_RMSE']:6.3f}")
    print(f"\n  -> {args.out.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
