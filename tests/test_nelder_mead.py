"""
Tests for the OptimizationWorkChain.
"""

import numpy as np


def test_nelder_mead(configure, submit_as_async):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """

    from sample_workchains import Norm
    from aiida_optimize.engines import NelderMead
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    tolerance = 1e-1
    result = WorkflowFactory('optimize.optimize').run(
        engine=NelderMead,
        engine_kwargs=ParameterData(
            dict=dict(
                xtol=tolerance,
                ftol=tolerance,
                simplex=[[0.], [1.]],
                result_key='result'
            )
        ),
        calculation_workchain=Norm,
    )
    assert 'calculation_uuid' in result
    assert np.isclose(
        load_node(result['calculation_uuid'].value).out.result.value,
        0.,
        atol=tolerance
    )
    assert np.isclose(result['optimizer_result'].value, 0, atol=tolerance)


def test_nelder_mead_submit(configure_with_daemon, wait_for):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """

    from sample_workchains import Norm
    from aiida_optimize.engines import NelderMead
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.work.run import submit
    tolerance = 0.5
    pid = submit(
        WorkflowFactory('optimize.optimize'),
        engine=NelderMead,
        engine_kwargs=ParameterData(
            dict=dict(
                xtol=tolerance,
                ftol=tolerance,
                simplex=[[0., 1.], [1., 1.], [1., 0.]],
                result_key='result'
            )
        ),
        calculation_workchain=Norm,
    ).pid
    wait_for(pid)
    calc = load_node(pid)
    assert np.isclose(
        load_node(calc.out.calculation_uuid.value).out.result.value,
        0.,
        atol=tolerance
    )
    assert np.isclose(calc.out.optimizer_result.value, 0, atol=tolerance)
