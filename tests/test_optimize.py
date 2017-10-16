import numpy as np


def test_optimize_run(configure_with_daemon):
    from aiida_optimize import TestWorkChain
    result = TestWorkChain.run()['result']
    assert np.isclose(result.value, 0, atol=1e-3)
