"""
Tests for the OptimizationWorkChain.
"""

import numpy as np


def test_bisect(configure, submit_as_async):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain, with the Bisection engine.
    """

    from echo_workchain import Echo
    from aiida_optimize.engines import Bisection
    from aiida.orm import WorkflowFactory
    from aiida.orm.data.parameter import ParameterData
    tolerance = 1e-1
    result = WorkflowFactory('optimize.optimize').run(
        engine=Bisection,
        engine_kwargs=ParameterData(
            dict=dict(lower=-1, upper=1, tol=tolerance)
        ),
        calculation_workchain=Echo
    )
    assert np.isclose(result['calculation_result'].value, 0, atol=tolerance)
    assert np.isclose(result['optimizer_result'].value, 0, atol=tolerance)
