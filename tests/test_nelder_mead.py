# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the OptimizationWorkChain.
"""

import numpy as np
import pytest

from aiida_optimize.engines import NelderMead


@pytest.mark.parametrize(
    [
        "func_workchain_name",
        "simplex",
        "xtol_input",
        "ftol_input",
        "xtol",
        "ftol",
        "x_exact",
        "f_exact",
    ],
    (
        ["Norm", [[0.0], [1.0]], None, None, np.inf, np.inf, [0.0], 0.0],
        ["Norm", [[0.0], [1.0]], 1e-1, None, 1e-1, np.inf, [0.0], 0.0],
        ["Norm", [[0.0], [1.0]], None, 1e-1, np.inf, 1e-1, [0.0], 0.0],
        ["Norm", [[0.0], [1.0]], 1e-1, 1e-1, 1e-1, 1e-1, [0.0], 0.0],
        [
            "rosenbrock",
            [[1.2, 0.9], [1.0, 2.0], [2.0, 1.0]],
            1e-1,
            1e-1,
            0.63,
            1e-1,
            [1.0, 1.0],
            0.0,
        ],
        ["sin_list", [[-np.pi / 2 + 1e-3], [np.pi]], 1e-1, 1e-1, 1e-1, 1e-1, [-np.pi / 2], -1.0],
        ["sin_list", [[-np.pi / 2 - 1e-3], [np.pi]], 1e-1, 1e-1, 1e-1, 1e-1, [-np.pi / 2], -1.0],
    ),
)
def test_nelder_mead(  # pylint: disable=too-many-arguments
    check_optimization,
    func_workchain_name,
    simplex,
    xtol_input,
    ftol_input,
    xtol,
    ftol,
    x_exact,
    f_exact,
):
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """

    check_optimization(
        engine=NelderMead,
        engine_kwargs=dict(
            simplex=simplex,
            xtol=xtol_input,
            ftol=ftol_input,
        ),
        func_workchain_name=func_workchain_name,
        xtol=xtol,
        ftol=ftol,
        x_exact=x_exact,
        f_exact=f_exact,
        output_port_names=[
            "engine_outputs__last_simplex",
        ],
    )


def test_nelder_mead_max_iter(check_error):
    """
    Test the OptimizationWorkChain with the Nelder-Mead engine in the case
    when the set `max_iter` is reached. This triggers the 202 exut status.
    """

    check_error(
        engine=NelderMead,
        engine_kwargs=dict(
            simplex=[[1.2, 0.9], [1.0, 2.0], [2.0, 1.0]],
            xtol=1e-1,
            ftol=1e-1,
            max_iter=10,
        ),
        func_workchain_name="rosenbrock",
        exit_status=202,
        output_port_names=[
            "engine_outputs__last_simplex",
        ],
    )
