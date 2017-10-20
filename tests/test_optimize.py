import pytest
import numpy as np


def test_optimize_run(configure, submit_as_async):
    # os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    from echo_workchain import Echo
    from aiida_optimize import OptimizationWorkChain
    result = OptimizationWorkChain.run(calculation_workchain=Echo)['result']
    assert np.isclose(result.value, 0, atol=1e-2)
