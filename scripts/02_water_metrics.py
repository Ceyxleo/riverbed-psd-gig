#!/usr/bin/env python3
"""Water-security variables per sample, for the reference and each function.

Chain, all driven by D84 except the Fredle Index (Methods, "Water variable
calculations"):

    tau_b   = rho g H S                          bed shear stress
    u*      = sqrt(tau_b / rho)                  shear velocity
    n       = k D84^(1/6)                        Strickler roughness, k = 0.0342
    U       = (1/n) R^(2/3) S^(1/2)              bankfull velocity
    Re      = U H / nu                           channel Reynolds number
    Re*     = u* D84 / nu                        shear (bed-particle) Reynolds number
    tau*    = tau_b / ((rho_s - rho) g D84)      Shields stress
    q_b     = 8 (tau* - tau*_c)^1.5 sqrt(g R D84^3)   Meyer-Peter Muller, tau*_c = 0.047
    tau_c   = theta_c (rho_s - rho) g D84        critical shear stress, theta_c = 0.045
    h_100   = (n Q100 / B sqrt(S))^(3/5)         100-year flood stage, wide channel
    FI      = sqrt(D16 D84) / sqrt(D75 / D25)    Fredle Index

NOTE: U, and therefore Re, depend on D84 through the Strickler roughness, so
neither is independent of grain size, and Re* contains D84 directly.

Bedload is defined only where tau* exceeds tau*_c; samples below the threshold of
motion carry no bedload prediction and drop out of the Fig. 5 bedload column.

    python scripts/03_water_metrics.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

G = 9.81
RHO_WATER = 1000.0
RHO_SEDIMENT = 2650.0
NU = 1e-6
STRICKLER_K = 0.0342
THETA_CRITICAL_MPM = 0.047      # Meyer-Peter Muller / Garcia (2008), for bedload
THETA_CRITICAL_BT = 0.045       # Berenbrock & Tranmer (2008), for critical shear stress

VARIANTS = ("reference", "GIG_2p", "Lognormal", "Weibull")
FUNCTIONS = ("GIG_2p", "Lognormal", "Weibull")

FIG5_ROWS = [
    ("D16", "D16_", ""),
    ("D50", "D50_", ""),
    ("D84", "D84_", ""),
    ("bedload_transport", "bedload_transport_", "_m2_s"),
    ("critical_shear_stress", "critical_shear_stress_", "_Pa"),
    ("flood_stage_100yr", "flood_stage_", "_m"),
    ("fredle_index", "fredle_index_", ""),
]


def positive(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr) & (arr > 0)


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    slope = pd.to_numeric(out["SLOPE"], errors="coerce").to_numpy(float)
    area = pd.to_numeric(out["BANKFULL_XSEC_AREA"], errors="coerce").to_numpy(float)
    width = pd.to_numeric(out["BANKFULL_WIDTH"], errors="coerce").to_numpy(float)
    depth = pd.to_numeric(out["BANKFULL_DEPTH"], errors="coerce").to_numpy(float)
    q100 = pd.to_numeric(out["Q100"], errors="coerce").to_numpy(float)

    geometry_ok = positive(slope) & positive(area) & positive(width) & positive(depth)
    tau_b = np.where(geometry_ok, RHO_WATER * G * depth * slope, np.nan)
    u_shear = np.where(geometry_ok, np.sqrt(tau_b / RHO_WATER), np.nan)
    hydraulic_radius = np.where(geometry_ok, area / (width + 2.0 * depth), np.nan)

    out["bed_shear_stress_Pa"] = tau_b
    out["shear_velocity_m_s"] = u_shear
    out["hydraulic_radius_m"] = hydraulic_radius

    for variant in VARIANTS:
        d_m = pd.to_numeric(out[f"D84_{variant}"], errors="coerce").to_numpy(float) / 1000.0
        d_ok = positive(d_m)
        both = geometry_ok & d_ok

        n_manning = np.where(d_ok, STRICKLER_K * np.power(d_m, 1.0 / 6.0), np.nan)
        velocity = np.where(both, (1.0 / n_manning) * np.power(hydraulic_radius, 2.0 / 3.0)
                            * np.sqrt(slope), np.nan)
        theta = np.where(both, tau_b / ((RHO_SEDIMENT - RHO_WATER) * G * d_m), np.nan)

        bedload = np.full(len(out), np.nan)
        moving = both & (theta > THETA_CRITICAL_MPM)
        bedload[moving] = (
            8.0 * np.power(theta[moving] - THETA_CRITICAL_MPM, 1.5)
            * np.sqrt(G * ((RHO_SEDIMENT - RHO_WATER) / RHO_WATER) * np.power(d_m[moving], 3.0))
        )
        bedload = np.round(bedload, 6)

        out[f"manning_n_{variant}"] = n_manning
        out[f"bankfull_velocity_{variant}_m_s"] = velocity
        out[f"Re_{variant}"] = np.where(both, velocity * depth / NU, np.nan)
        out[f"Re_shear_{variant}"] = np.where(both, u_shear * d_m / NU, np.nan)
        out[f"shields_stress_{variant}"] = theta
        out[f"bedload_transport_{variant}_m2_s"] = bedload
        out[f"critical_shear_stress_{variant}_Pa"] = np.where(
            d_ok, THETA_CRITICAL_BT * (RHO_SEDIMENT - RHO_WATER) * G * d_m, np.nan)
        out[f"flood_stage_{variant}_m"] = np.where(
            both & positive(q100),
            np.power(n_manning * q100 / width / np.sqrt(slope), 3.0 / 5.0), np.nan)

        d16 = out[f"D16_{variant}"].to_numpy(float)
        d25 = out[f"D25_{variant}"].to_numpy(float)
        d75 = out[f"D75_{variant}"].to_numpy(float)
        d84 = out[f"D84_{variant}"].to_numpy(float)
        valid = positive(d16) & positive(d25) & positive(d75) & positive(d84)
        out[f"fredle_index_{variant}"] = np.where(
            valid, np.sqrt(d16 * d84) / np.sqrt(d75 / d25), np.nan)

    return out.replace([np.inf, -np.inf], np.nan)


def fig5_subset(frame: pd.DataFrame, prefix: str, suffix: str, name: str):
    reference = f"{prefix}reference{suffix}"
    columns = [reference] + [f"{prefix}{f}{suffix}" for f in FUNCTIONS]
    subset = frame.loc[frame[columns].notna().all(axis=1)]
    dropped_zero = 0
    if name == "bedload_transport":
        minimum = subset[reference].min()
        dropped_zero = int((subset[reference] == minimum).sum())
        subset = subset.loc[subset[reference] != minimum]
    subset = subset.loc[subset[reference] != 0]
    return subset, reference, dropped_zero


def rmsre_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, prefix, suffix in FIG5_ROWS:
        subset, reference, dropped = fig5_subset(frame, prefix, suffix, name)
        row = {"variable": name, "n_samples": len(subset),
               "n_sites": subset["site_no"].nunique(), "n_dropped_zero_reference": dropped}
        for function in FUNCTIONS:
            relative = (subset[reference] - subset[f"{prefix}{function}{suffix}"]) / subset[reference]
            row[f"RMSRE_{function}_percent"] = float(np.sqrt(np.mean(relative ** 2)) * 100)
            row[f"meanSRE_{function}"] = float(np.mean(relative ** 2) * 100)
            row[f"sdSRE_{function}"] = float(np.std(relative ** 2, ddof=1) * 100)
        for other in ("Lognormal", "Weibull"):
            row[f"reduction_vs_{other}_percent"] = float(
                (1 - row["RMSRE_GIG_2p_percent"] / row[f"RMSRE_{other}_percent"]) * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dvalues", type=Path, default=ROOT / "data" / "dvalues.csv")
    parser.add_argument("--nhd", type=Path,
                        default=ROOT / "data" / "station_nhdplus_attributes.csv")
    parser.add_argument("--q100", type=Path,
                        default=ROOT / "data" / "station_q100_logpearson3.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "water")
    parser.add_argument("--tables", type=Path, default=ROOT / "outputs" / "tables")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    args.out.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)

    dvalues = pd.read_csv(args.dvalues, dtype={"site_no": str})
    before = len(dvalues)
    # Only the percentiles these variables consume: D16/D50/D84 for Fig. 5 and the
    # hydraulics, D25/D75 for the Fredle Index. D05 and D95 are carried in the
    # percentile table for tail diagnostics and must not gate sample selection.
    used = [f"{label}_{variant}" for label in ("D16", "D25", "D50", "D75", "D84")
            for variant in VARIANTS]
    dvalues = dvalues.dropna(subset=[c for c in used if c in dvalues.columns])
    print(f"samples with a complete set of D16/D25/D50/D75/D84 values: "
          f"{len(dvalues):,} of {before:,} ({before - len(dvalues)} dropped)")

    nhd = pd.read_csv(args.nhd, dtype={"site_no": str}).drop_duplicates("site_no")
    q100 = pd.read_csv(args.q100, dtype={"site_no": str}).drop_duplicates("site_no")
    frame = dvalues.merge(nhd, on="site_no", how="left").merge(q100, on="site_no", how="left")

    frame = compute(frame)
    out_path = args.out / "sample_water_metrics.csv"
    frame.to_csv(out_path, index=False)
    print(f"wrote {len(frame):,} rows x {len(frame.columns)} columns -> "
          f"{out_path.relative_to(ROOT)}")

    table = rmsre_table(frame)
    table.to_csv(args.tables / "figure5_rmsre_matrix.csv", index=False)
    print("\nFigure 5 -- RMSRE (%), lower is better\n")
    print(f"{'variable':22s} {'n':>7s} {'sites':>6s} " +
          "".join(f"{f:>12s}" for f in FUNCTIONS) + f"{'vs Logn':>9s}{'vs Weib':>9s}")
    for _, row in table.iterrows():
        print(f"{row['variable']:22s} {row['n_samples']:7,} {row['n_sites']:6,} " +
              "".join(f"{row[f'RMSRE_{f}_percent']:12.3f}" for f in FUNCTIONS) +
              f"{row['reduction_vs_Lognormal_percent']:8.1f}%{row['reduction_vs_Weibull_percent']:8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
