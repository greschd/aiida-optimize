import os

import pytest
import numpy as np


@pytest.fixture(scope='session')
def set_pythonpath():
    """
    Set the PYTHONPATH to include the current working directory. Note that this needs to run before 'configure_with_daemon'.
    """
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))


def test_optimize_run(set_pythonpath, configure_with_daemon):
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

    from echo_workchain import Echo
    from aiida_optimize import OptimizationWorkChain
    result = OptimizationWorkChain.run(calculation_workchain=Echo)['result']
    assert np.isclose(result.value, 0, atol=1e-3)
