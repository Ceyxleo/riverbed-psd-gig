#!/usr/bin/env python3
"""Populate `data/` from the data deposit.

The deposit holds the quality-controlled samples and the archived fitted
parameters; this repository holds only code and small reference tables, so the
bulk inputs are copied in once before anything else runs.

    python scripts/00_fetch_inputs.py --deposit ../psd-gig-data-v1.0.0

Nothing here reaches the network. Download the deposit from its Zenodo DOI (see
README.md) and point `--deposit` at the unpacked directory.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPOSIT = ROOT.parent / "psd-gig-data-v1.0.0"

FILES = [
    ("01_samples/usgs_psd_samples_qc.csv", "data/usgs_psd_samples_qc.csv"),
    ("01_samples/usgs_sample_statistics.csv", "data/usgs_sample_statistics.csv"),
    ("01_samples/usgs_station_metadata.csv", "data/usgs_station_metadata.csv"),
    ("02_fits/best_fit_by_sample.csv", "data/best_fit_by_sample.csv"),
    ("03_percentiles/dvalues_archived.csv", "data/dvalues_archived.csv"),
    ("04_hydraulics/station_nhdplus_attributes.csv", "data/station_nhdplus_attributes.csv"),
    ("04_hydraulics/station_q100_logpearson3.csv", "data/station_q100_logpearson3.csv"),
]
FIT_GLOB = "02_fits/fits_F*.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deposit", type=Path, default=DEFAULT_DEPOSIT)
    args = parser.parse_args()

    if not args.deposit.exists():
        parser.error(f"Deposit not found: {args.deposit}")

    (ROOT / "data" / "fitted_functions").mkdir(parents=True, exist_ok=True)
    copied = 0
    for relative, target in FILES:
        source = args.deposit / relative
        if not source.exists():
            print(f"  MISSING {relative}")
            continue
        shutil.copy2(source, ROOT / target)
        copied += 1
        print(f"  {target}")

    fits = sorted(args.deposit.glob(FIT_GLOB))
    for source in fits:
        shutil.copy2(source, ROOT / "data" / "fitted_functions" / source.name)
    print(f"  data/fitted_functions/  ({len(fits)} files)")

    print(f"\n{copied + len(fits)} files copied from {args.deposit}")
    if not (ROOT / "data" / "rhine" / "rhine_data_to_fit.csv").exists():
        print("\nLower Rhine input is absent, as expected -- it cannot be redistributed.")
        print("See data/rhine/README.md to build it from the 4TU.ResearchData record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
