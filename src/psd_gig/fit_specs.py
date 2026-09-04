"""The 25 candidate PSD functions and how each one is fitted.

Every function is grid-searched: for each sample the fit is repeated from a grid of
initial parameter values and the lowest-BIC result is kept. The grid is sized by
parameter count -- 50 guesses for the two-parameter functions, 8 for the
three-parameter ones -- so that all 25 are selected under one procedure.

`x_transform` says whether a function is fitted against D or against log2(D); it applies
to both modes. The remaining `base_*` fields describe the single-guess fit and are used
only by `--mode base`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import function_library as funcs


#: Tensor grids applied by parameter count.
GRID_2P: tuple[tuple[float, float, float], ...] = ((0.5, 3, 0.5), (-2, 3, 0.5))   # 5 x 10 = 50
GRID_3P: tuple[tuple[float, float, float], ...] = ((1, 3, 1), (1, 3, 1), (1, 3, 1))  # 2^3 = 8


@dataclass(frozen=True)
class GridSpec:
    init_ranges: tuple[tuple[float, float, float], ...]

    def initial_values(self) -> list[tuple[float, ...]]:
        values = [np.arange(start, stop, step).tolist() for start, stop, step in self.init_ranges]
        mesh = np.meshgrid(*values, indexing="ij")
        return [tuple(float(array.ravel()[idx]) for array in mesh) for idx in range(mesh[0].size)]

    def __len__(self) -> int:
        return len(self.initial_values())


@dataclass(frozen=True)
class FunctionSpec:
    number: int
    label: str
    func: Callable
    n_params: int
    grid: GridSpec
    x_transform: str = "linear"
    base_initial_guess: tuple[float, ...] | None = None
    base_retry_initial_guess: tuple[float, ...] | None = None

    @property
    def id(self) -> str:
        return f"F{self.number:02d}_{self.label}"

    @property
    def output_name(self) -> str:
        """The one output filename, in `fitted_functions/` or in `base_fits/`."""
        return f"fits_F{self.number:02d}_{self.label}.csv"


def _spec(
    number: int,
    label: str,
    func: Callable,
    n_params: int,
    *,
    grid: tuple[tuple[float, float, float], ...] | None = None,
    x_transform: str = "linear",
    base_guess: tuple[float, ...] | None = None,
    base_retry: tuple[float, ...] | None = None,
) -> FunctionSpec:
    if grid is None:
        grid = GRID_2P if n_params == 2 else GRID_3P
    return FunctionSpec(
        number=number,
        label=label,
        func=func,
        n_params=n_params,
        grid=GridSpec(init_ranges=grid),
        x_transform=x_transform,
        base_initial_guess=base_guess,
        base_retry_initial_guess=base_retry,
    )


_R2 = (0.5, 2.5)
_R3 = (0.5, 2.5, -0.5)

# Three functions carry a narrower grid than the default for their parameter count,
# because that is the grid the shipped CONUS fits were produced with (see README):
# GLH_p05 and GLH_p1 at 20 guesses, GIG_2p at 25.
FUNCTION_SPECS: tuple[FunctionSpec, ...] = (
    _spec(1, "Algeb", funcs.Algeb, 2, base_retry=_R2),
    _spec(2, "Erf_PL", funcs.Erf_PL, 2, base_retry=_R2),
    _spec(3, "Exp_PL", funcs.Exp_PL, 2, base_retry=_R2),
    _spec(4, "Gamma", funcs.Gamma, 2, base_retry=_R2),
    _spec(5, "GLH_p05", funcs.GLH_p05, 2, grid=((0.5, 3, 0.5), (1, 3, 0.5)),
          x_transform="log2", base_guess=(2, 1)),
    _spec(6, "GLH_p1", funcs.GLH_p1, 2, grid=((0.5, 3, 0.5), (1, 3, 0.5)),
          x_transform="log2", base_guess=(2, 1)),
    _spec(7, "Tanh", funcs.Tanh, 2, base_retry=_R2),
    _spec(8, "Log_exp", funcs.Log_exp, 2, base_retry=_R2),
    _spec(9, "Ln", funcs.Ln, 2, base_retry=_R2),
    _spec(10, "Lognormal", funcs.Lognormal, 2, base_retry=_R2),
    _spec(11, "LLaplace", funcs.LLaplace, 2, base_retry=_R2),
    _spec(12, "NIG", funcs.NIG, 2, base_guess=(1, 0.5)),
    _spec(13, "PL", funcs.PL, 2, base_retry=_R2),
    _spec(14, "GIG_2p", funcs.GIG_2p, 2, grid=((0.5, 3, 0.5), (0.5, 3, 0.5)), base_retry=_R2),
    _spec(15, "Weibull", funcs.Weibull, 2, base_retry=_R2),
    _spec(16, "GLH", funcs.GLH, 3, x_transform="log2", base_guess=_R3),
    _spec(17, "Logn_Weib", funcs.Logn_Weib, 3, base_retry=_R3),
    _spec(18, "Logn_PL", funcs.Logn_PL, 3, base_retry=_R3),
    _spec(19, "Logn_Tanh", funcs.Logn_Tanh, 3, base_retry=_R3),
    _spec(20, "LSLaplace", funcs.LSLaplace, 3, base_retry=_R3),
    _spec(21, "PL_Tanh", funcs.PL_Tanh, 3, base_retry=_R3),
    _spec(22, "pPL_Exp", funcs.pPL_Exp, 3, base_retry=_R3),
    _spec(23, "PL_omExp", funcs.PL_omExp, 3, base_retry=_R3),
    _spec(24, "PL_Weib", funcs.PL_Weib, 3, base_retry=_R3),
    _spec(25, "GIG_3p", funcs.GIG_3p, 3, base_retry=_R3),
)

FUNCTION_SPECS_BY_LABEL = {spec.label: spec for spec in FUNCTION_SPECS}
FUNCTION_SPECS_BY_NUMBER = {spec.number: spec for spec in FUNCTION_SPECS}

#: Functions whose shipped USGS fits come from the original maxfev = 1e6 grid runs
#: rather than from a re-run in this repository.
SHIPPED_FROM_ARCHIVE = (5, 6, 10, 12, 14, 15, 16, 18, 25)


def select_function_specs(selectors: list[str] | None) -> list[FunctionSpec]:
    if not selectors:
        return list(FUNCTION_SPECS)

    selected = []
    for selector in selectors:
        key = selector.strip()
        if not key:
            continue
        if key.isdigit():
            selected.append(FUNCTION_SPECS_BY_NUMBER[int(key)])
        else:
            selected.append(FUNCTION_SPECS_BY_LABEL[key])
    return selected
