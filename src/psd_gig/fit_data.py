"""Loading of PSD sample tables and their grain-diameter lookups.

Two input conventions are supported, selected in the config:

``size_lookup.kind: file``
    Column names are USGS parameter codes (``p80164`` ...) and the diameter for
    each code is read from a one-row wide CSV (``data/reference/sieve_sizes_all43.csv``).
    This is the USGS convention.

``size_lookup.kind: column_name``
    The diameter is encoded in the column name itself (``"0.5 mm"``, ``"31.5 mm"``).
    This is the convention of the Lower Rhine table.

``size_columns: auto`` selects every column matching ``size_pattern``. The USGS
fits published with the paper used *all* available sieve grades, not only the
twelve principal codes, so ``auto`` is the correct setting there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import resolve_project_path


def _select_size_columns(config: dict, data: pd.DataFrame) -> list[str]:
    configured = config["data"]["size_columns"]
    if configured != "auto":
        size_columns = list(configured)
        missing = [column for column in size_columns if column not in data.columns]
        if missing:
            raise ValueError(f"Configured size columns missing from input: {missing}")
        return size_columns

    pattern = re.compile(config["data"]["size_pattern"])
    size_columns = [column for column in data.columns if pattern.match(str(column))]
    if not size_columns:
        raise ValueError(f"size_pattern {pattern.pattern!r} matched no columns")
    return size_columns


def load_input_data(config: dict) -> tuple[pd.DataFrame, list[str]]:
    input_path = resolve_project_path(config, config["paths"]["input_data_file"])
    data = pd.read_csv(input_path, dtype={"site_no": str}, low_memory=False)
    data["sample_ID"] = data["sample_ID"].astype(str)

    size_columns = _select_size_columns(config, data)

    limit_samples = config.get("runtime", {}).get("limit_samples")
    if limit_samples:
        data = data.iloc[: int(limit_samples)].copy()

    return data, size_columns


def load_diameter_lookup(config: dict, size_columns: list[str]) -> pd.DataFrame:
    """Return a frame indexed by size-column name with a single ``d`` column, in mm."""
    lookup = config["data"]["size_lookup"]
    kind = lookup["kind"]

    if kind == "column_name":
        pattern = re.compile(r"([0-9]*\.?[0-9]+)")
        rows = []
        for column in size_columns:
            match = pattern.search(str(column))
            if match is None:
                raise ValueError(f"Cannot read a diameter from column name {column!r}")
            rows.append({"size_code": str(column), "d": float(match.group(1))})
        return pd.DataFrame(rows).set_index("size_code")

    if kind == "file":
        path = resolve_project_path(config, lookup["path"])
        table = pd.read_csv(path).T.reset_index()
        table = table.rename(columns={"index": "size_code", 0: "d"})
        table["size_code"] = table["size_code"].astype(str)
        table["d"] = table["d"].astype(float)
        table = table.set_index("size_code")
        missing = [column for column in size_columns if column not in table.index]
        if missing:
            raise ValueError(f"No diameter in {path.name} for: {missing}")
        return table

    raise ValueError(f"Unknown size_lookup.kind: {kind!r}")


def write_data_inventory(
    config: dict, output_path: Path, data: pd.DataFrame, size_columns: list[str]
) -> None:
    input_path = resolve_project_path(config, config["paths"]["input_data_file"])
    rows = [
        {
            "name": "input_data_file",
            "path": str(input_path),
            "rows": len(data),
            "columns": len(data.columns),
            "size_columns": len(size_columns),
            "size_column_names": " ".join(map(str, size_columns)),
        }
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
