"""
Tests for the OptimizationWorkChain.
"""

from __future__ import print_function

import numpy as np


def test_bisect(configure):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain, with the Bisection engine.
    """

    from sample_workchains import Echo
    from aiida_optimize.engines import Bisection
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.work.launch import run
    tolerance = 1e-1
    result = run(
        WorkflowFactory('optimize.optimize'),
        engine=Bisection,
        engine_kwargs=ParameterData(dict=dict(lower=-1, upper=1, tol=tolerance)),
        calculation_workchain=Echo
    )
    assert 'calculation_uuid' in result
    assert np.isclose(
        load_node(result['calculation_uuid'].value).out.result.value, 0., atol=tolerance
    )
    assert np.isclose(result['optimizer_result'].value, 0, atol=tolerance)


def test_bisect_submit(configure_with_daemon, wait_for):  # pylint: disable=unused-argument
    """
    Test that submits the OptimizationWorkChain with the Bisection engine.
    """

    from sample_workchains import Echo
    from aiida_optimize.engines import Bisection
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.work.launch import submit
    tolerance = 1e-1
    print('submit')
    pk = submit(
        WorkflowFactory('optimize.optimize'),
        engine=Bisection,
        engine_kwargs=ParameterData(dict=dict(lower=-1, upper=1, tol=tolerance)),
        calculation_workchain=Echo
    ).pk
    print('wait')
    wait_for(pk)
    print('load')
    calc = load_node(pk)
    assert np.isclose(
        load_node(calc.out.calculation_uuid.value).out.result.value, 0., atol=tolerance
    )
    assert np.isclose(calc.out.optimizer_result.value, 0, atol=tolerance)
