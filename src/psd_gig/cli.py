from __future__ import annotations

import argparse

from .fit_specs import FUNCTION_SPECS, EXPENSIVE_FUNCTIONS, select_function_specs
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
        "--mode",
        choices=["grid", "base"],
        default="grid",
        help="grid: fit from every initial guess and keep the lowest BIC (default). "
             "base: one guess per sample, written to a separate base_fits/ folder.",
    )
    fit.add_argument(
        "--skip-expensive", action="store_true",
        help="Omit the nine functions that are expensive to fit because their CDFs are "
             "evaluated by numerical integration (F05, F06, F10, F12, F14, F15, F16, F18, F25).",
    )
    fit.add_argument(
        "--jobs", type=int, default=1,
        help="Fit this many functions concurrently, one process each (default 1).",
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
            print(f"{spec.number:02d} {spec.label:12s} p={spec.n_params} "
                  f"grid={len(spec.grid)} guesses")
        return

    if args.command == "fit-functions":
        selectors = args.functions
        if args.skip_expensive:
            chosen = select_function_specs(selectors)
            selectors = [str(s.number) for s in chosen if s.number not in EXPENSIVE_FUNCTIONS]
        outputs = run_fit_workflow(
            args.config,
            function_selectors=selectors,
            mode=args.mode,
            jobs=args.jobs,
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

