from __future__ import annotations

import argparse

from .fit_specs import FUNCTION_SPECS
from .fit_workflow import run_fit_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated PSD reproduction workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit-functions",
        help="Fit PSD functions from usgs_data_to_fit.csv and write fitted-function outputs.",
    )
    fit.add_argument("--config", default="configs/fit_workflow.yaml")
    fit.add_argument(
        "--functions",
        nargs="*",
        help="Optional function labels or numbers, for example: 1 Algeb Lognormal 25.",
    )
    fit.add_argument(
        "--stages",
        nargs="+",
        choices=["base", "grid", "final"],
        default=["base", "grid", "final"],
        help="Workflow stages to run.",
    )
    fit.add_argument("--overwrite", action="store_true", help="Overwrite existing generated outputs.")
    fit.add_argument("--limit-samples", type=int, help="Override config runtime.limit_samples.")

    subparsers.add_parser("list-functions", help="Print available function numbers and labels.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-functions":
        for spec in FUNCTION_SPECS:
            suffix = " grid" if spec.grid else " base"
            print(f"{spec.number:02d} {spec.label}{suffix}")
        return

    if args.command == "fit-functions":
        outputs = run_fit_workflow(
            args.config,
            function_selectors=args.functions,
            stages=args.stages,
            overwrite=args.overwrite or None,
            limit_samples=args.limit_samples,
        )
        print("Generated fit workflow outputs:")
        for label, path in outputs.items():
            print(f"- {label}: {path}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

