"""
Tests for the OptimizationWorkChain.
"""

from __future__ import print_function

import pytest


@pytest.mark.parametrize(
    ['func_workchain_name', 'simplex', 'xtol', 'ftol', 'x_exact', 'f_exact', 'check_x'],
    (
        ['Norm', [[0.], [1.]], 1e-1, 1e-1, [0.], 0., True],
        ['rosenbrock', [[1.2, 0.9], [1., 2.], [2., 1.]], 1e-1, 1e-1, [1., 1.], 0., False],
    )  # pylint: disable=too-many-arguments
)
def test_nelder_mead(
    check_optimization,
    func_workchain_name,
    simplex,
    xtol,
    ftol,
    x_exact,
    f_exact,
    check_x,
):
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """
    from aiida_optimize.engines import NelderMead

    check_optimization(
        engine=NelderMead,
        engine_kwargs=dict(
            simplex=simplex,
            xtol=xtol,
            ftol=ftol,
        ),
        func_workchain_name=func_workchain_name,
        x_exact=x_exact,
        f_exact=f_exact,
        check_x=check_x,
    )
