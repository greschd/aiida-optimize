"""
Tests for the OptimizationWorkChain.
"""

from __future__ import print_function

import numpy as np
import pytest


@pytest.mark.parametrize(
    ['func_workchain_name', 'simplex', 'xtol', 'ftol', 'x_exact', 'f_exact'],
    (
        # [
        #     'Norm',
        #     [[0.], [1.]],
        #     1e-1,
        #     1e-1,
        #     [0.],
        #     0.
        # ],
        ['rosenbrock', [[1.2, 0.9], [1., 2.], [2., 1.]], 1e-2, 1e-1, [1., 1.], 0.],
    )  # pylint: disable=too-many-arguments
)
def test_nelder_mead(
    configure,  # pylint: disable=unused-argument
    run_optimization,
    func_workchain_name,
    simplex,
    xtol,
    ftol,
    x_exact,
    f_exact,
):
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """
    from aiida.orm import load_node
    from aiida_optimize.engines import NelderMead

    import sample_workchains
    func_workchain = getattr(sample_workchains, func_workchain_name)

    result = run_optimization(
        engine=NelderMead,
        engine_kwargs=dict(
            simplex=simplex,
            xtol=xtol,
            ftol=ftol,
            result_key='result',
        ),
        func_workchain=func_workchain,
    )

    assert 'calculation_uuid' in result
    assert np.isclose(result['optimizer_result'].value, f_exact, atol=ftol)
    calc = load_node(result['calculation_uuid'].value)
    assert np.allclose(list(calc.get_inputs_dict()['x']), x_exact, atol=xtol)
