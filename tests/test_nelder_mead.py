"""
Tests for the OptimizationWorkChain.
"""

from __future__ import print_function

import numpy as np
import pytest


@pytest.mark.parametrize(
    [
        'func_workchain_name', 'simplex', 'xtol_input', 'ftol_input', 'xtol', 'ftol', 'x_exact',
        'f_exact'
    ],
    (
        ['Norm', [[0.], [1.]], 1e-1, 1e-1, 1e-1, 1e-1, [0.], 0.],
        ['rosenbrock', [[1.2, 0.9], [1., 2.], [2., 1.]], 1e-1, 1e-1, 0.4, 1e-1, [1., 1.], 0.],
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
    from aiida_optimize.engines import NelderMead

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
