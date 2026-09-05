#!/usr/bin/env python3
"""Supplementary Table S5: paired tests of the error reduction achieved by GIG.

For each water-security variable, the squared relative error of a sample under
function f is

    SRE_f = ((X_reference - X_f) / X_reference)^2

and a paired Wilcoxon signed-rank test is run on (SRE_other - SRE_GIG) with the
one-sided alternative that the difference is greater than zero, i.e. that GIG has
the smaller error.

The table reports only that test. Error magnitudes are not repeated here: RMSRE is
what Fig. 5 plots, and the mean of SRE is identically (RMSRE in %)^2 / 100, which is
the ambiguity the submitted "SRE, %" heading created.

Two further things this fixes relative to the submitted table:

* the statistic column was headed "t-statistics", which is wrong for a signed-rank
  test. The Wilcoxon W is reported.
* no statistic was actually given, only p-values. With between 12,930 and 19,762
  pairs every p-value is far below any threshold, so the win rate and the
  matched-pairs rank-biserial correlation are reported: they are what distinguishes
  the comparisons. The win rate counts how often GIG wins, the rank-biserial
  correlation weights those wins by how large they are.

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
OTHERS = ("Lognormal", "Weibull")


def subset(frame: pd.DataFrame, prefix: str, suffix: str, name: str):
    reference = f"{prefix}reference{suffix}"
    columns = [reference] + [f"{prefix}{f}{suffix}" for f in FUNCTIONS]
    rows = frame.loc[frame[columns].notna().all(axis=1)]
    if name == "bedload_transport":
        rows = rows.loc[rows[reference] != rows[reference].min()]
    return rows.loc[rows[reference] != 0], reference


def squared_relative_error(reference: pd.Series, prediction: pd.Series) -> pd.Series:
    return ((reference - prediction) / reference) ** 2


def format_p(p: float) -> str:
    return "< 1 x 10^-300" if p == 0 else f"{p:.0e}".replace("e-", " x 10^-")


def render(table: pd.DataFrame, path: Path) -> None:
    """The paper-ready Table S5, one row per comparison."""
    lines = [
        "**Supplementary Table S5 | Paired Wilcoxon signed-rank tests of the reduction in "
        "error achieved by GIG.**",
        "",
        "| Variable | Comparison | Samples, n | Tied pairs, n | W | p (one-sided) | "
        "GIG lower, % | r_rb |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        for other in OTHERS:
            lines.append(
                f"| {row['variable']} | GIG vs {other} | {row['n_samples']:,} | "
                f"{row[f'n_ties_vs_{other}']:,} | {row[f'wilcoxon_W_vs_{other}']:,.0f} | "
                f"{format_p(row[f'wilcoxon_p_one_sided_vs_{other}'])} | "
                f"{row[f'gig_lower_error_share_vs_{other}_percent']:.1f} | "
                f"{row[f'rank_biserial_vs_{other}']:.3f} |")

    low = table[[f"gig_lower_error_share_vs_{o}_percent" for o in OTHERS]].to_numpy()
    rb = table[[f"rank_biserial_vs_{o}" for o in OTHERS]].to_numpy()
    lines += [
        "",
        "For each water-security variable, the squared relative error of a fitted "
        "distribution against the reference value, SRE_f = ((X_ref - X_f) / X_ref)^2, was "
        "compared pairwise between GIG and each of Lognormal and Weibull. W is the sum of "
        "the ranks of the differences SRE_other - SRE_GIG that are positive, under the "
        "one-sided alternative that GIG has the smaller error; it is computed over the "
        "decided pairs, that is the n samples less the tied pairs. Ties are structural "
        "rather than coincidental: where GIG and the comparison function are both in a "
        "sample's best-fit set, the reference is their mean and the two lie the same "
        "distance from it. \"GIG lower\" is the percentage of decided pairs in which GIG's "
        "error is the smaller, and r_rb is the matched-pairs rank-biserial correlation, "
        "(W+ - W-) / (W+ + W-), which weights those wins by their size. With between "
        f"{table['n_samples'].min():,} and {table['n_samples'].max():,} pairs every p-value "
        "falls far below any conventional threshold, so the win rate and r_rb are the "
        "quantities that distinguish the comparisons. Error magnitudes are given in Fig. 5.",
        "",
        f"Summary for the main text: GIG has the lower error in {low.min():.0f}-{low.max():.0f}% "
        f"of decided pairs, with rank-biserial correlations of {rb.min():.2f}-{rb.max():.2f}.",
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

        for other in OTHERS:
            difference = (errors[other] - errors["GIG_2p"]).to_numpy()
            w, p = stats.wilcoxon(difference, zero_method="wilcox", alternative="greater")
            wins = int(np.sum(difference > 0))
            losses = int(np.sum(difference < 0))
            ties = int(np.sum(difference == 0))
            decided = wins + losses
            row[f"wilcoxon_W_vs_{other}"] = float(w)
            row[f"wilcoxon_p_one_sided_vs_{other}"] = float(p)
            row[f"n_wins_vs_{other}"] = wins
            row[f"n_losses_vs_{other}"] = losses
            # Exact ties are structural, not coincidental: when GIG and the other
            # function are both in a sample's best-fit set, the reference is their
            # mean, so both sit the same distance from it. `zero_method="wilcox"`
            # discards them before ranking, so W and the win rate are both over the
            # decided pairs -- which is why the table reports the tie count.
            row[f"n_ties_vs_{other}"] = ties
            row[f"n_decided_vs_{other}"] = decided
            row[f"gig_lower_error_share_vs_{other}_percent"] = (
                float(wins / decided * 100) if decided else float("nan"))
            # Matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-).
            # This uses the ranks, so it reflects how large GIG's wins are, not just
            # how many there are -- the win-rate column already counts those.
            total_rank_sum = decided * (decided + 1) / 2
            row[f"rank_biserial_vs_{other}"] = float(
                (2 * w - total_rank_sum) / total_rank_sum) if total_rank_sum else float("nan")
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "table_S5_significance.csv", index=False)

    render(table, args.out / "table_S5_paper.md")
    print((args.out / "table_S5_paper.md").read_text())
    print(f"  -> {(args.out / 'table_S5_significance.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
