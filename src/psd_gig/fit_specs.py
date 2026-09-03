from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import function_library as funcs


@dataclass(frozen=True)
class GridSpec:
    grid_output_name: str
    final_output_name: str
    init_ranges: tuple[tuple[float, float, float], ...]

    def initial_values(self) -> list[tuple[float, ...]]:
        values = [np.arange(start, stop, step).tolist() for start, stop, step in self.init_ranges]
        mesh = np.meshgrid(*values, indexing="ij")
        return [tuple(float(array.ravel()[idx]) for array in mesh) for idx in range(mesh[0].size)]


@dataclass(frozen=True)
class FunctionSpec:
    number: int
    label: str
    func: Callable
    n_params: int
    base_output_name: str
    final_output_name: str
    base_x_transform: str = "linear"
    base_initial_guess: tuple[float, ...] | None = None
    base_retry_initial_guess: tuple[float, ...] | None = None
    grid: GridSpec | None = None

    @property
    def id(self) -> str:
        return f"F{self.number:02d}_{self.label}"


def _grid(
    grid_output_name: str,
    final_output_name: str,
    *ranges: tuple[float, float, float],
) -> GridSpec:
    return GridSpec(
        grid_output_name=grid_output_name,
        final_output_name=final_output_name,
        init_ranges=tuple(ranges),
    )


FUNCTION_SPECS: tuple[FunctionSpec, ...] = (
    FunctionSpec(1, "Algeb", funcs.Algeb, 2, "F01_Algeb.csv", "F01_Algeb.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(2, "Erf_PL", funcs.Erf_PL, 2, "F02_Erf_PL.csv", "F02_Erf_PL.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(3, "Exp_PL", funcs.Exp_PL, 2, "F03_Exp_PL.csv", "F03_Exp_PL.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(4, "Gamma", funcs.Gamma, 2, "F04_Gamma.csv", "F04_Gamma.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(
        5,
        "GLH_p05",
        funcs.GLH_p05,
        2,
        "F05_GLH_p05_base.csv",
        "F05_GLH_p05.csv",
        base_x_transform="log2",
        base_initial_guess=(2, 1),
        grid=_grid("f05_glh_p05_gridsearch_all.csv", "F05_GLH_p05.csv", (0.5, 3, 0.5), (1, 3, 0.5)),
    ),
    FunctionSpec(
        6,
        "GLH_p1",
        funcs.GLH_p1,
        2,
        "F06_GLH_p1_base.csv",
        "F06_GLH_p1.csv",
        base_x_transform="log2",
        base_initial_guess=(2, 1),
        grid=_grid("f06_glh_p1_gridsearch_all.csv", "F06_GLH_p1.csv", (0.5, 3, 0.5), (1, 3, 0.5)),
    ),
    FunctionSpec(7, "Tanh", funcs.Tanh, 2, "F07_Tanh.csv", "F07_Tanh.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(8, "Log_exp", funcs.Log_exp, 2, "F08_Log_exp.csv", "F08_Log_exp.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(9, "Ln", funcs.Ln, 2, "F09_Ln.csv", "F09_Ln.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(
        10,
        "Lognormal",
        funcs.Lognormal,
        2,
        "F10_Lognormal_base.csv",
        "F10_Lognormal_grids_well.csv",
        base_retry_initial_guess=(0.5, 2.5),
        grid=_grid("f10_lognormal_gridsearch_all.csv", "F10_Lognormal_grids_well.csv", (0.5, 3, 0.5), (-2, 3, 0.5)),
    ),
    FunctionSpec(11, "LLaplace", funcs.LLaplace, 2, "F11_LLaplace.csv", "F11_LLaplace.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(
        12,
        "NIG",
        funcs.NIG,
        2,
        "F12_NIG_base.csv",
        "F12_NIG.csv",
        base_initial_guess=(1, 0.5),
        grid=_grid("f12_NIG_gridsearch_all.csv", "F12_NIG.csv", (0.5, 3, 0.5), (-2, 3, 0.5)),
    ),
    FunctionSpec(13, "PL", funcs.PL, 2, "F13_PL.csv", "F13_PL.csv", base_retry_initial_guess=(0.5, 2.5)),
    FunctionSpec(
        14,
        "GIG_2p",
        funcs.GIG_2p,
        2,
        "F14_GIG_2p_base.csv",
        "F14_GIG_2p_grids_well.csv",
        base_retry_initial_guess=(0.5, 2.5),
        grid=_grid("f14_gig_gridsearch_all.csv", "F14_GIG_2p_grids_well.csv", (0.5, 3, 0.5), (0.5, 3, 0.5)),
    ),
    FunctionSpec(
        15,
        "Weibull",
        funcs.Weibull,
        2,
        "F15_Weibull_base.csv",
        "F15_Weibull_grids_well.csv",
        base_retry_initial_guess=(0.5, 2.5),
        grid=_grid("f15_Weibull_gridsearch_all.csv", "F15_Weibull_grids_well.csv", (0.5, 3, 0.5), (-2, 3, 0.5)),
    ),
    FunctionSpec(
        16,
        "GLH",
        funcs.GLH,
        3,
        "F16_GLH_base.csv",
        "F16_GLH_grids_well.csv",
        base_x_transform="log2",
        base_initial_guess=(0.5, 2.5, -0.5),
        grid=_grid("f16_GLH_gridsearch_all.csv", "F16_GLH_grids_well.csv", (1, 3, 1), (1, 3, 1), (1, 3, 1)),
    ),
    FunctionSpec(17, "Logn_Weib", funcs.Logn_Weib, 3, "F17_Logn_Weib.csv", "F17_Logn_Weib.csv", base_retry_initial_guess=(0.5, 2.5, -0.5)),
    FunctionSpec(
        18,
        "Logn_PL",
        funcs.Logn_PL,
        3,
        "F18_Logn_PL_base.csv",
        "F18_Logn_PL_grids_well.csv",
        base_retry_initial_guess=(0.5, 2.5, -0.5),
        grid=_grid("f18_Logn_PL_gridsearch_all.csv", "F18_Logn_PL_grids_well.csv", (1, 3, 1), (1, 3, 1), (1, 3, 1)),
    ),
    FunctionSpec(19, "Logn_Tanh", funcs.Logn_Tanh, 3, "F19_Logn_Tanh.csv", "F19_Logn_Tanh.csv", base_retry_initial_guess=(0.5, 2.5, -0.5)),
    FunctionSpec(20, "LSLaplace", funcs.LSLaplace, 3, "F20_LSLaplace.csv", "F20_LSLaplace.csv", base_retry_initial_guess=(0.5, 2.5, -0.5)),
    FunctionSpec(21, "PL_Tanh", funcs.PL_Tanh, 3, "F21_PL_Tanh.csv", "F21_PL_Tanh.csv", base_retry_initial_guess=(0.5, 2.5, -0.5)),
    FunctionSpec(22, "pPL_Exp", funcs.pPL_Exp, 3, "F22_pPL_Exp.csv", "F22_pPL_Exp.csv", base_retry_initial_guess=(0.5, 2.5, -0.5)),
    FunctionSpec(23, "PL_omExp", funcs.PL_omExp, 3, "F23_PL_omExp.csv", "F23_PL_omExp.csv", base_retry_initial_guess=(0.5, 2.5, -0.5)),
    FunctionSpec(24, "PL_Weib", funcs.PL_Weib, 3, "F24_PL_Weib.csv", "F24_PL_Weib.csv", base_retry_initial_guess=(0.5, 2.5, -0.5)),
    FunctionSpec(
        25,
        "GIG_3p",
        funcs.GIG_3p,
        3,
        "F25_GIG_3p_base.csv",
        "F25_GIG_3p_grids_well.csv",
        base_retry_initial_guess=(0.5, 2.5, -0.5),
        grid=_grid("f25_GIG_3p_gridsearch_all.csv", "F25_GIG_3p_grids_well.csv", (1, 3, 1), (1, 3, 1), (1, 3, 1)),
    ),
)

FUNCTION_SPECS_BY_LABEL = {spec.label: spec for spec in FUNCTION_SPECS}
FUNCTION_SPECS_BY_NUMBER = {spec.number: spec for spec in FUNCTION_SPECS}


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

