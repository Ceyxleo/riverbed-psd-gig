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
  19,762 every p-value is far below any threshold, so the matched-pairs rank-biserial
  correlation and the win rate are reported alongside; they are what actually
  distinguish the comparisons. The win rate counts how often GIG wins, the
  rank-biserial correlation weights those wins by how large they are.

    python scripts/05_significance.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

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


def render(table: pd.DataFrame, path: Path) -> None:
    """The paper-ready Table S5, in the shape recommended to replace the submitted one."""
    names = {"GIG_2p": "GIG", "Lognormal": "Lognormal", "Weibull": "Weibull"}
    lines = [
        "**Table S5. Accuracy of the water-security variables under GIG, Lognormal and "
        "Weibull, and paired tests of GIG's advantage.**",
        "",
        "| Variable | n samples (sites) | RMSRE (%) | | | Median abs. rel. error (%) | | |",
        "|---|---|---|---|---|---|---|---|",
        "| | | " + " | ".join(names.values()) + " | " + " | ".join(names.values()) + " |",
    ]
    for _, row in table.iterrows():
        cells = [f"{row[f'RMSRE_{f}_percent']:.2f}" for f in FUNCTIONS]
        cells += [f"{row[f'medianABSRE_{f}_percent']:.2f}" for f in FUNCTIONS]
        lines.append(f"| {row['variable']} | {row['n_samples']:,} ({row['n_sites']:,}) | "
                     + " | ".join(cells) + " |")

    lines += [
        "",
        "| Variable | Contrast | GIG has the lower error | Tied | Rank-biserial r |",
        "|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        for other in ("Lognormal", "Weibull"):
            wins, losses = row[f"n_wins_vs_{other}"], row[f"n_losses_vs_{other}"]
            lines.append(
                f"| {row['variable']} | vs {other} | "
                f"{row[f'gig_lower_error_share_vs_{other}_percent']:.1f}% "
                f"({wins:,} of {wins + losses:,} decided) | "
                f"{row[f'n_ties_vs_{other}']:,} | "
                f"{row[f'rank_biserial_vs_{other}']:+.3f} |")

    worst = table[[f"wilcoxon_p_one_sided_vs_{o}" for o in ("Lognormal", "Weibull")]].to_numpy().max()
    lines += [
        "",
        "One-sided Wilcoxon signed-rank test on the paired difference "
        "SRE_other - SRE_GIG, where SRE = ((X_ref - X_f)/X_ref)^2 and X_ref is the mean over "
        f"the sample's best-fit set. Every p < {worst:.0e}; the W statistics and exact p-values "
        "are in `table_S5_significance.csv`. With n this large the p-values separate nothing, "
        "so the win rate and the rank-biserial correlation are given instead: the first counts "
        "how often GIG wins, the second weights those wins by size. Ties are exact and "
        "structural -- when both functions are in a sample's best-fit set the reference lies "
        "midway between them -- so the win rate is taken over decided pairs. RMSRE is the "
        "quantity plotted in Fig. 5; the median is given alongside because the mean squared "
        "error is dominated by a small number of extreme samples.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            # The mean is tail-dominated -- on bedload the SD is 50-86x the mean -- so
            # the typical sample needs its own number.
            absolute = np.sqrt(errors[f] / 100) * 100
            row[f"medianABSRE_{f}_percent"] = float(absolute.median())
            row[f"p90ABSRE_{f}_percent"] = float(absolute.quantile(0.90))
        rows.append(row)

        for other in ("Lognormal", "Weibull"):
            difference = (errors[other] - errors["GIG_2p"]).to_numpy()
            w, p = stats.wilcoxon(difference, zero_method="wilcox", alternative="greater")
            wins = int(np.sum(difference > 0))
            losses = int(np.sum(difference < 0))
            ties = int(np.sum(difference == 0))
            nonzero = wins + losses
            row[f"wilcoxon_W_vs_{other}"] = float(w)
            row[f"wilcoxon_p_one_sided_vs_{other}"] = float(p)
            row[f"n_wins_vs_{other}"] = wins
            row[f"n_losses_vs_{other}"] = losses
            # Exact ties are structural, not coincidental: when GIG and the other
            # function are both in a sample's best-fit set, the reference is their
            # mean, so both sit the same distance from it. Counting ties as losses
            # would understate the win rate, so the share is over decided pairs.
            row[f"n_ties_vs_{other}"] = ties
            row[f"gig_lower_error_share_vs_{other}_percent"] = (
                float(wins / nonzero * 100) if nonzero else float("nan"))
            # Matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-).
            # This uses the ranks, so it reflects how large GIG's wins are, not just
            # how many there are -- the win-rate column already counts those.
            total_rank_sum = nonzero * (nonzero + 1) / 2
            row[f"rank_biserial_vs_{other}"] = float(
                (2 * w - total_rank_sum) / total_rank_sum) if total_rank_sum else float("nan")

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "table_S5_significance.csv", index=False)

    render(table, args.out / "table_S5_paper.md")
    print((args.out / "table_S5_paper.md").read_text())

    print(f"\n  -> {(args.out / 'table_S5_significance.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
