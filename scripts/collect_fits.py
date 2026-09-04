#!/usr/bin/env python3
"""Assemble the 25 canonical CONUS fits in `data/fitted_functions/` and record provenance.

All 25 are grid-searched. Sixteen are produced by this repository at maxfev = 20,000;
nine are taken from the original runs at maxfev = 1e6 because their CDFs are evaluated
by numerical integration and re-fitting them would take weeks (see README, "Fitting").
This script puts both groups in one folder under one naming convention and writes
`provenance.csv` saying where each file came from.

    python scripts/collect_fits.py                       # after `make fits`
    python scripts/collect_fits.py --archive <dir>       # non-default archive location
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psd_gig.fit_specs import FUNCTION_SPECS, SHIPPED_FROM_ARCHIVE  # noqa: E402

#: Filenames in the original archive, which predate this repository's convention.
ARCHIVE_NAMES = {
    5: "F05_GLH_p05.csv", 6: "F06_GLH_p1.csv", 10: "F10_Lognormal_grids_well.csv",
    12: "F12_NIG.csv", 14: "F14_GIG_2p_grids_well.csv", 15: "F15_Weibull_grids_well.csv",
    16: "F16_GLH_grids_well.csv", 18: "F18_Logn_PL_grids_well.csv",
    25: "F25_GIG_3p_grids_well.csv",
}
KEEP = ["site_no", "sample_ID", "fitted_A", "fitted_B", "fitted_C", "RMSE", "R2", "AIC", "BIC"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refit", type=Path,
                        default=ROOT / "outputs" / "fit_usgs" / "fitted_functions")
    parser.add_argument("--archive", type=Path, default=ROOT.parent.parent / "Lingbo" / "fitted_functions")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fitted_functions")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in FUNCTION_SPECS:
        shipped = spec.number in SHIPPED_FROM_ARCHIVE
        source = (args.archive / ARCHIVE_NAMES[spec.number]) if shipped \
            else (args.refit / spec.output_name)
        if not source.exists():
            print(f"  MISSING {source}")
            continue
        frame = pd.read_csv(source, dtype={"site_no": str}).rename(columns={"R^2": "R2"})
        frame = frame[[c for c in KEEP if c in frame.columns]]
        frame.to_csv(args.out / spec.output_name, index=False)
        rows.append({
            "function_number": spec.number, "function": spec.label,
            "n_params": spec.n_params, "n_initial_guesses": len(spec.grid),
            "x_transform": spec.x_transform,
            "maxfev": 1_000_000 if shipped else 20_000,
            "provenance": "original grid run" if shipped else "re-fitted by this repository",
            "source": str(source), "n_samples": len(frame),
            "median_BIC": round(float(frame["BIC"].median()), 4),
        })
        print(f"  F{spec.number:02d} {spec.label:11s} {'shipped' if shipped else 'refit ':7s} "
              f"{len(frame):,} rows  median BIC {frame['BIC'].median():8.3f}")

    provenance = pd.DataFrame(rows)
    provenance.to_csv(args.out / "provenance.csv", index=False)
    n_ship = int((provenance["provenance"] == "original grid run").sum())
    print(f"\n{len(provenance)} functions in {args.out.relative_to(ROOT)}: "
          f"{len(provenance) - n_ship} re-fitted here, {n_ship} shipped.")
    return 0 if len(provenance) == 25 else 1


if __name__ == "__main__":
    raise SystemExit(main())
