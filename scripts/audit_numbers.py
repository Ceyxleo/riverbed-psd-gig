#!/usr/bin/env python3
"""Check every number the manuscript asserts against the regenerated outputs.

Each row prints the claim as published, the value this pipeline produces, and a
status:

    OK        agrees within tolerance
    MISMATCH  disagrees -- the manuscript needs correcting, or the pipeline does
    SKIP      the input needed for this check has not been produced yet

Exits non-zero if anything is MISMATCH, so it can gate a release.

    python scripts/audit_numbers.py
    python scripts/audit_numbers.py --section fig5 --section rhine
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
OUT = ROOT / "outputs"

RESULTS: list[tuple[str, str, str, str, str]] = []


def _fmt(value) -> str:
    """Counts read as counts; everything else keeps four significant figures."""
    number = float(value)
    if number == int(number) and abs(number) >= 1000:
        return f"{int(number):,}"
    return f"{number:.4g}"


def record(section: str, claim: str, published, computed, tolerance=0.05,
           note: str = "") -> None:
    if computed is None:
        RESULTS.append((section, claim, str(published), "-", "SKIP"))
        return
    if isinstance(published, str):
        status = "OK" if published == str(computed) else "MISMATCH"
        RESULTS.append((section, claim, published, str(computed), status))
        return
    ok = abs(float(computed) - float(published)) <= tolerance
    RESULTS.append((section, claim, _fmt(published), _fmt(computed),
                    "OK" if ok else "MISMATCH"))
    if note and not ok:
        RESULTS.append((section, f"    note: {note}", "", "", ""))


def read(path: Path, **kwargs):
    return pd.read_csv(path, **kwargs) if path.exists() else None


# --------------------------------------------------------------------- checks

def check_counts() -> None:
    stats_table = read(ROOT / "data" / "usgs_sample_statistics.csv", dtype={"site_no": str})
    if stats_table is None:
        record("counts", "quality-controlled samples", 19765, None)
        return
    record("counts", "quality-controlled samples", 19765, len(stats_table), 0)
    record("counts", "stations", 2114, stats_table["site_no"].nunique(), 0)

    fig5 = read(OUT / "tables" / "figure5_rmsre_matrix.csv")
    if fig5 is None:
        return
    rows = fig5.set_index("variable")
    record("counts", "samples with D values (Fig. 5)", 19763,
           rows.loc["D84", "n_samples"], 0,
           note="one sample (25839) dropped: its fitted GIG CDF is not numerically "
                "evaluable, so no percentile can be solved from it")
    record("counts", "bedload samples", 14567, rows.loc["bedload_transport", "n_samples"], 0)
    record("counts", "bedload sites", 1466, rows.loc["bedload_transport", "n_sites"], 0)
    record("counts", "flood-stage samples", 12930, rows.loc["flood_stage_100yr", "n_samples"], 0)
    record("counts", "flood-stage sites", 684, rows.loc["flood_stage_100yr", "n_sites"], 0)
    record("counts", "Methods: flood-stage samples", 11448,
           rows.loc["flood_stage_100yr", "n_samples"], 0,
           note="Methods says 606 stations / 11,448 samples; Fig. 5 and the data say 684 / 12,930")


def check_best_fit() -> None:
    frame = read(ROOT / "data" / "usgs_sample_statistics.csv", dtype={"site_no": str})
    if frame is None:
        record("best-fit", "GIG best-fit share (%)", 70, None)
        return
    n = len(frame)
    best = frame["best_function(s)"].astype(str)
    for label, name, published, unique_published in (
            ("GIG_2p", "GIG", 70, 24), ("Lognormal", "Lognormal", 48, None),
            ("Weibull", "Weibull", 38, None)):
        included = best.str.contains(label, regex=False)
        record("best-fit", f"{name} best-fit share (%)", published,
               included.mean() * 100, 0.5)
        if unique_published is not None:
            unique = included & (frame["num_functions"] == 1)
            record("best-fit", f"{name} unique best-fit share (%)", unique_published,
                   unique.mean() * 100, 0.5)
            record("best-fit", f"{name} tied best-fit share (%)", 46,
                   (included.sum() - unique.sum()) / n * 100, 0.5)

    classes = pd.cut(frame["skew_FW"], [-np.inf, -0.1, 0.1, np.inf],
                     labels=["coarse", "symmetric", "fine"])
    for name, published in (("coarse", 37), ("symmetric", 28), ("fine", 35)):
        record("best-fit", f"{name}-skewed share (%)", published,
               (classes == name).mean() * 100, 0.5)
    for name, expected in (("coarse", (89, 62, 9)), ("symmetric", (64, 68, 22)),
                           ("fine", (55, 16, 82))):
        selection = (classes == name).to_numpy()
        for label, display, published in zip(("GIG_2p", "Lognormal", "Weibull"),
                                             ("GIG", "Lognormal", "Weibull"), expected):
            record("best-fit", f"{name}-skewed, {display} best-fit (%)", published,
                   best[selection].str.contains(label, regex=False).mean() * 100, 0.6)


def check_bic_tables() -> None:
    s2 = read(OUT / "tables" / "table_S2_median_metrics.csv")
    fig4 = read(OUT / "tables" / "figure4_delta_bic_summary.csv")
    if s2 is None:
        record("BIC", "GIG_3p median BIC", 16.7, None)
        return
    medians = s2.set_index("function")["median_BIC"]
    # For four of these six the main text was taken from an earlier fitting run,
    # not from the deposited fits that Table S2 and everything downstream use.
    superseded = {"Logn_PL", "Lognormal", "GLH", "Weibull"}
    for function, published in (("GIG_3p", 16.7), ("GIG_2p", 19.5), ("Logn_PL", 19.8),
                                ("Lognormal", 21.3), ("GLH", 21.9), ("Weibull", 24.5)):
        note = ("main text predates the deposited fits; Table S2 is correct"
                if function in superseded else "")
        record("BIC", f"{function} median BIC", published, medians[function], 0.05, note)
    # Table S2's outlier column, for the nine functions whose fits are shipped
    # unchanged. The other sixteen were re-fitted, so their counts are expected to move.
    published_outliers = {"GLH_p05": 121, "GLH_p1": 137, "Lognormal": 167, "NIG": 212,
                          "GIG_2p": 235, "Weibull": 266, "GLH": 366, "Logn_PL": 434,
                          "GIG_3p": 397}
    if "n_BIC_outliers" in s2.columns:
        counts = s2.set_index("function")["n_BIC_outliers"]
        for function, published in published_outliers.items():
            record("BIC", f"{function} BIC outliers (Table S2)", published,
                   counts[function], 0,
                   note="Tukey fliers in BOTH tails; low BIC is a good fit, so the "
                        "total mixes poor and unusually good samples")

    record("BIC", "GIG_2p ranks ahead of Logn_PL", "yes",
           "yes" if medians["GIG_2p"] < medians["Logn_PL"] else "no")
    record("BIC", "  gap GIG_2p vs Logn_PL (BIC units, negligible if < 2)", 2.0,
           abs(medians["GIG_2p"] - medians["Logn_PL"]), 2.0)
    if fig4 is not None:
        share = fig4.set_index("function")["lowest_BIC_share_percent"]
        record("BIC", "GIG_3p lowest-BIC share (%)", 27.5, share["GIG_3p"], 0.1)
        record("BIC", "GIG_2p lowest-BIC share (%)", 15.0, share["GIG_2p"], 0.1)


def check_fig5() -> None:
    fig5 = read(OUT / "tables" / "figure5_rmsre_matrix.csv")
    if fig5 is None:
        record("fig5", "D84 RMSRE, GIG (%)", 4.1, None)
        return
    rows = fig5.set_index("variable")
    # The published values were computed from percentiles solved by a grid scan of
    # the fitted CDF. This pipeline inverts the CDF analytically, which lands on
    # each target percentile to ~1e-11 instead of overshooting it, so Lognormal's
    # and Weibull's percentile errors come out slightly lower and GIG's relative
    # advantage slightly smaller. Where a cell disagrees below, the manuscript
    # value is the one to update.
    method = "percentiles now solved analytically rather than by a grid scan"
    published = {
        "D16": (5.5, 7.3, 10.5), "D50": (3.3, 5.5, 7.0), "D84": (4.1, 11.6, 8.7),
    }
    for variable, values in published.items():
        for function, value in zip(("GIG_2p", "Lognormal", "Weibull"), values):
            record("fig5", f"{variable} RMSRE, {function} (%)", value,
                   rows.loc[variable, f"RMSRE_{function}_percent"], 0.06, method)
    reductions = {"bedload_transport": (47.2, 54.3), "critical_shear_stress": (64.7, 52.9),
                  "flood_stage_100yr": (58.8, 62.3), "fredle_index": (29.1, 42.5)}
    for variable, (versus_lognormal, versus_weibull) in reductions.items():
        record("fig5", f"{variable}: reduction vs Lognormal (%)", versus_lognormal,
               rows.loc[variable, "reduction_vs_Lognormal_percent"], 0.1, method)
        record("fig5", f"{variable}: reduction vs Weibull (%)", versus_weibull,
               rows.loc[variable, "reduction_vs_Weibull_percent"], 0.1, method)
    record("fig5", "critical shear stress RMSRE equals D84 RMSRE", "yes",
           "yes" if abs(rows.loc["critical_shear_stress", "RMSRE_GIG_2p_percent"]
                        - rows.loc["D84", "RMSRE_GIG_2p_percent"]) < 1e-9 else "no")


def check_flood_stage() -> None:
    metrics = read(OUT / "water" / "sample_water_metrics.csv",
                   dtype={"site_no": str}, low_memory=False)
    if metrics is None:
        record("flood", "GIG flood-stage MAE (m)", 0.012, None)
        return
    columns = ["flood_stage_reference_m"] + [f"flood_stage_{f}_m"
                                             for f in ("GIG_2p", "Lognormal", "Weibull")]
    subset = metrics.loc[metrics[columns].notna().all(axis=1)]
    reference = subset["flood_stage_reference_m"]
    rmse = {}
    for function, mae_published, rmse_published in (("GIG_2p", 0.012, 0.029),
                                                    ("Lognormal", 0.031, 0.089),
                                                    ("Weibull", 0.044, 0.090)):
        error = reference - subset[f"flood_stage_{function}_m"]
        rmse[function] = float(np.sqrt(np.mean(error ** 2)))
        record("flood", f"Methods: {function} MAE (m)", mae_published,
               float(np.mean(np.abs(error))), 0.002)
        record("flood", f"Methods: {function} RMSE (m)", rmse_published, rmse[function], 0.005)
    record("flood", "Discussion: flood-stage reduction (m)", 0.06,
           rmse["Lognormal"] - rmse["GIG_2p"], 0.005,
           note="stale; recomputed on the current 12,930-sample set")


def check_correlations() -> None:
    metrics = read(OUT / "water" / "sample_water_metrics.csv",
                   dtype={"site_no": str}, low_memory=False)
    fit_path = ROOT / "data" / "fitted_functions" / "fits_F14_GIG_2p.csv"
    if metrics is None or not fit_path.exists():
        record("fig3", "rho_s(eta, Re*)", 0.85, None)
        return
    frame = metrics[["sample_ID", "site_no", "Re_reference", "Re_shear_reference"]].rename(
        columns={"Re_reference": "Re", "Re_shear_reference": "Re_star"})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["Re", "Re_star"])
    frame = frame[(frame["Re"] > 0) & (frame["Re_star"] > 0)]
    record("fig3", "hydraulically complete samples", 19596, len(frame), 0)
    fit = pd.read_csv(fit_path, usecols=["sample_ID", "fitted_A", "fitted_B"])
    merged = frame.merge(fit, on="sample_ID", how="inner")
    for column, name, published in (("fitted_A", "eta", 0.85), ("fitted_B", "beta", -0.77)):
        clean = merged.dropna(subset=[column])
        rho, _ = stats.spearmanr(clean[column], clean["Re_star"])
        record("fig3", f"Spearman rho({name}, Re*)", published, rho, 0.01)
        pearson, _ = stats.pearsonr(clean[column], clean["Re_star"])
        RESULTS.append(("fig3", f"    Pearson r({name}, Re*) for comparison", "-",
                        f"{pearson:+.3f}", ""))
    for column, name, published in (("fitted_A", "eta", -0.16), ("fitted_B", "beta", 0.19)):
        clean = merged.dropna(subset=[column])
        rho, _ = stats.spearmanr(clean[column], clean["Re"])
        record("fig3", f"Spearman rho({name}, Re)", published, rho, 0.01)


def check_rhine() -> None:
    summary = read(OUT / "rhine" / "rhine_function_summary.csv")
    samples = read(OUT / "rhine" / "rhine_sample_table.csv", dtype={"sample_ID": str})
    if summary is None or samples is None:
        for name, value in (("GIG", 10.07), ("Lognormal", 13.99), ("Weibull", 9.16)):
            record("rhine", f"{name} median BIC", value, None)
        return
    record("rhine", "samples", 67, len(samples), 0)
    medians = summary.set_index("function")["median_BIC"]
    for function, name, published in (("GIG_2p", "GIG", 10.07), ("Lognormal", "Lognormal", 13.99),
                                      ("Weibull", "Weibull", 9.16)):
        if function in medians.index:
            record("rhine", f"{name} median BIC", published, medians[function], 0.05)
    best = samples["best_function(s)"].fillna("").astype(str)
    record("rhine", "GIG best-fit share (%)", 52, best.str.contains("GIG_2p").mean() * 100, 0.6)
    record("rhine", "Weibull best-fit share (%)", 52,
           best.str.contains("Weibull").mean() * 100, 0.6)
    counts = samples["skew_class"].value_counts()
    record("rhine", "main text: 48 samples are FINE-skewed", 48,
           int(counts.get("fine-skewed", 0)), 0,
           note="the 48 are COARSE-skewed under the paper's own Folk-Ward convention")
    record("rhine", "  of which coarse-skewed", 48, int(counts.get("coarse-skewed", 0)), 0)
    record("rhine", "near-symmetric samples", 18, int(counts.get("near-symmetric", 0)), 0)


def check_table_s5() -> None:
    table = read(OUT / "tables" / "table_S5_significance.csv")
    if table is None:
        record("tableS5", "bedload mean SRE, GIG", 1.48, None)
        return
    rows = table.set_index("variable")
    published = {"Bedload transport": (1.48, 5.30, 7.09),
                 "Critical shear stress": (0.17, 1.34, 0.75),
                 "100-year flood stage": (0.001, 0.008, 0.010),
                 "Fredle Index": (0.22, 0.45, 0.68)}
    method = "percentiles now solved analytically rather than by a grid scan"
    for variable, values in published.items():
        for function, value in zip(("GIG_2p", "Lognormal", "Weibull"), values):
            record("tableS5", f"{variable}: mean SRE, {function}", value,
                   rows.loc[variable, f"meanSRE_{function}"], 0.006, method)


SECTIONS = {
    "counts": check_counts, "best-fit": check_best_fit, "bic": check_bic_tables,
    "fig5": check_fig5, "flood": check_flood_stage, "fig3": check_correlations,
    "rhine": check_rhine, "tableS5": check_table_s5,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", action="append", choices=list(SECTIONS))
    parser.add_argument("--out", type=Path, default=OUT / "audit_manuscript_numbers.csv")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    for name in (args.section or list(SECTIONS)):
        SECTIONS[name]()

    print(f"{'section':10s} {'claim':52s} {'paper':>10s} {'computed':>11s}  status")
    print("-" * 96)
    for section, claim, published, computed, status in RESULTS:
        marker = {"OK": "ok", "MISMATCH": "MISMATCH <<", "SKIP": "skip", "": ""}[status]
        print(f"{section:10s} {claim:52s} {published:>10s} {computed:>11s}  {marker}")

    table = pd.DataFrame(RESULTS, columns=["section", "claim", "published",
                                           "computed", "status"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    counts = table["status"].value_counts()
    print("-" * 96)
    print(f"{int(counts.get('OK', 0))} agree, {int(counts.get('MISMATCH', 0))} mismatched, "
          f"{int(counts.get('SKIP', 0))} skipped  ->  {args.out.relative_to(ROOT)}")
    return 1 if counts.get("MISMATCH", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
