"""Shared figure-output handling.

Figures are written as PNG by default. `--format pdf` or `--format both` is the
hook for vector output when a journal asks for it.
"""
from __future__ import annotations

from pathlib import Path


def add_format_argument(parser) -> None:
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png",
                        help="Output format (default: png).")


def save_figure(fig, out_prefix, fmt: str = "png", dpi: int = 600, **kwargs) -> list[Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if fmt in {"png", "both"}:
        path = out_prefix.with_suffix(".png")
        fig.savefig(path, dpi=dpi, **kwargs)
        written.append(path)
    if fmt in {"pdf", "both"}:
        path = out_prefix.with_suffix(".pdf")
        fig.savefig(path, **kwargs)
        written.append(path)
    for path in written:
        print(path)
    return written
