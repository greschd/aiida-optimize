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
        'func_workchain_name', 'simplex', 'xtol_input', 'ftol_input', 'xtol', 'ftol', 'x_exact',
        'f_exact'
    ],
    (
        ['Norm', [[0.], [1.]], None, None, np.inf, np.inf, [0.], 0.],
        ['Norm', [[0.], [1.]], 1e-1, None, 1e-1, np.inf, [0.], 0.],
        ['Norm', [[0.], [1.]], None, 1e-1, np.inf, 1e-1, [0.], 0.],
        ['Norm', [[0.], [1.]], 1e-1, 1e-1, 1e-1, 1e-1, [0.], 0.],
        ['rosenbrock', [[1.2, 0.9], [1., 2.], [2., 1.]], 1e-1, 1e-1, 0.63, 1e-1, [1., 1.], 0.],
        ['sin_list', [[-np.pi / 2 + 1e-3], [np.pi]], 1e-1, 1e-1, 1e-1, 1e-1, [-np.pi / 2], -1.],
        ['sin_list', [[-np.pi / 2 - 1e-3], [np.pi]], 1e-1, 1e-1, 1e-1, 1e-1, [-np.pi / 2], -1.],
    )  # pylint: disable=too-many-arguments
)
def test_nelder_mead(
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
    )
