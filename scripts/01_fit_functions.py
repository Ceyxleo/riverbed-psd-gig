from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from psd_gig.cli import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in {"fit-functions", "list-functions"}:
        main(args)
    else:
        main(["fit-functions", *args])
