"""
Tests for the OptimizationWorkChain.
"""

import numpy as np


def test_parameter_sweep(configure, submit_as_async):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain with the ParameterSweep engine.
    """

    from sample_workchains import Echo
    from aiida_optimize.engines import ParameterSweep
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    result = WorkflowFactory('optimize.optimize').run(
        engine=ParameterSweep,
        engine_kwargs=ParameterData(
            dict=dict(
                result_key='result',
                parameters=[{
                    'x': x
                } for x in np.linspace(-2, 2, 10)]
            )
        ),
        calculation_workchain=Echo
    )
    assert 'calculation_uuid' in result
    assert load_node(result['calculation_uuid'].value).out.result.value == -2
    assert result['optimizer_result'].value == -2


def test_parameter_sweep_submit(configure_with_daemon, wait_for):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain with the ParameterSweep engine.
    """

    from sample_workchains import Echo
    from aiida_optimize.engines import ParameterSweep
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.work.run import submit
    pid = submit(
        WorkflowFactory('optimize.optimize'),
        engine=ParameterSweep,
        engine_kwargs=ParameterData(
            dict=dict(
                result_key='result',
                parameters=[{
                    'x': x
                } for x in np.linspace(-2, 2, 10)]
            )
        ),
        calculation_workchain=Echo,
    ).pid
    wait_for(pid)
    calc = load_node(pid)
    assert 'calculation_uuid' in calc.get_outputs_dict()
    assert calc.out.optimizer_result.value == -2
    assert load_node(calc.out.calculation_uuid.value).out.result.value == -2


def test_parameter_sweep_add(configure, submit_as_async):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain with the ParameterSweep engine and an additional fixed input.
    """

    from sample_workchains import Add
    from aiida_optimize.engines import ParameterSweep
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.base import Float
    from aiida.orm.data.parameter import ParameterData
    result = WorkflowFactory('optimize.optimize').run(
        engine=ParameterSweep,
        engine_kwargs=ParameterData(
            dict=dict(
                result_key='result',
                parameters=[{
                    'x': x
                } for x in np.linspace(-2, 2, 10)]
            )
        ),
        calculation_workchain=Add,
        calculation_inputs={
            'y': Float(1.)
        }
    )
    assert 'calculation_uuid' in result
    assert load_node(result['calculation_uuid'].value).out.result.value == -1
    assert result['optimizer_result'].value == -1
