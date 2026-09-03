"""Loaders that hand the figure scripts the frames they expect.

The deposit splits the original monolithic ``DATA_final_all_optfuncs.csv`` into a
sample table, a statistics table and a station table. The figure scripts want a
single joined frame, so that join lives here rather than being repeated in each
one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_samples(with_sieve_columns: bool = False) -> pd.DataFrame:
    """Sample statistics joined to station metadata, one row per sample.

    ``with_sieve_columns`` additionally joins the 43 measured sieve percentages,
    which only the PSD-curve figures need.
    """
    stats = pd.read_csv(DATA / "usgs_sample_statistics.csv", dtype={"site_no": str})
    stations = pd.read_csv(DATA / "usgs_station_metadata.csv", dtype={"site_no": str})
    frame = stats.merge(stations.drop(columns=["stream_type"], errors="ignore"),
                        on="site_no", how="left")
    if with_sieve_columns:
        samples = pd.read_csv(DATA / "usgs_psd_samples_qc.csv",
                              dtype={"site_no": str}, low_memory=False)
        sieve = [c for c in samples.columns if c.startswith("p") and c[1:].isdigit()]
        frame = frame.merge(
            samples[["sample_ID", "MonitoringLocationIdentifier",
                     "ActivityStartDate"] + sieve],
                            on="sample_ID", how="left")
    frame["sample_ID"] = frame["sample_ID"].astype(int)
    return frame


def load_fit(function_number: int, label: str) -> pd.DataFrame:
    path = DATA / "fitted_functions" / f"fits_F{function_number:02d}_{label}.csv"
    fit = pd.read_csv(path, dtype={"site_no": str})
    fit["sample_ID"] = fit["sample_ID"].astype(int)
    return fit


def load_all_fits() -> dict[str, pd.DataFrame]:
    from .fit_specs import FUNCTION_SPECS
    return {spec.label: load_fit(spec.number, spec.label) for spec in FUNCTION_SPECS}


def load_water_metrics() -> pd.DataFrame:
    path = ROOT / "outputs" / "water" / "sample_water_metrics.csv"
    return pd.read_csv(path, dtype={"site_no": str}, low_memory=False)


def sieve_sizes() -> pd.DataFrame:
    """Long table: column_name, size_mm, sorted by size."""
    wide = pd.read_csv(DATA / "reference" / "sieve_sizes_all43.csv")
    table = pd.DataFrame({"column_name": list(wide.columns),
                          "size_mm": wide.iloc[0].astype(float).to_numpy()})
    return table.sort_values("size_mm").reset_index(drop=True)
