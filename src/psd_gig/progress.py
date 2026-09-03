from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import json
import os
import time


FIELDNAMES = [
    "run_id",
    "timestamp_utc",
    "event",
    "stage",
    "function_number",
    "function",
    "completed_samples",
    "total_samples",
    "percent_samples",
    "completed_curve_fits",
    "total_curve_fits",
    "percent_curve_fits",
    "current_site_no",
    "current_sample_ID",
    "elapsed_seconds",
    "message",
]


@dataclass
class ProgressTracker:
    logs_dir: Path
    enabled: bool = True
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    def __post_init__(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.logs_dir / "progress_events.jsonl"
        self.latest_path = self.logs_dir / "progress_latest.csv"
        self.by_function_path = self.logs_dir / "progress_by_function.csv"
        self.started_at = time.monotonic()
        self._latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        if self.enabled:
            self.events_path.write_text("", encoding="utf-8")

    def record(
        self,
        event: str,
        *,
        stage: str = "",
        function_number: int | str = "",
        function: str = "",
        completed_samples: int | str = "",
        total_samples: int | str = "",
        completed_curve_fits: int | str = "",
        total_curve_fits: int | str = "",
        current_site_no: str = "",
        current_sample_ID: str = "",
        message: str = "",
        append_event: bool = True,
    ) -> None:
        if not self.enabled:
            return

        row = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "stage": stage,
            "function_number": function_number,
            "function": function,
            "completed_samples": completed_samples,
            "total_samples": total_samples,
            "percent_samples": _percent(completed_samples, total_samples),
            "completed_curve_fits": completed_curve_fits,
            "total_curve_fits": total_curve_fits,
            "percent_curve_fits": _percent(completed_curve_fits, total_curve_fits),
            "current_site_no": current_site_no,
            "current_sample_ID": current_sample_ID,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 2),
            "message": message,
        }

        if append_event:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        self._write_rows(self.latest_path, [row])
        if stage and function:
            self._latest_by_key[(stage, function)] = row
            self._write_rows(self.by_function_path, list(self._latest_by_key.values()))

    def _write_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        tmp_path = path.with_name(f"{path.name}.{self.run_id}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
        tmp_path.replace(path)


def _percent(completed: int | str, total: int | str) -> str:
    try:
        completed_int = int(completed)
        total_int = int(total)
    except (TypeError, ValueError):
        return ""
    if total_int == 0:
        return ""
    return round(completed_int / total_int * 100, 2)
