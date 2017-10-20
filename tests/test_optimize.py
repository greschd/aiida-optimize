import pytest
import numpy as np


def test_optimize_run(configure, submit_as_async):
    from echo_workchain import Echo
    from aiida_optimize.engines import Bisection
    from aiida_optimize.workchain import OptimizationWorkChain
    from aiida.orm.data.parameter import ParameterData
    TOLERANCE = 1e-1
    result = OptimizationWorkChain.run(
        engine=Bisection,
        engine_kwargs=ParameterData(
            dict=dict(lower=-1, upper=1, tol=TOLERANCE)
        ),
        calculation_workchain=Echo
    )['result']
    assert np.isclose(result.value, 0, atol=TOLERANCE)
