"""
Tests for the OptimizationWorkChain.
"""

from __future__ import print_function

import numpy as np


def test_nelder_mead(configure):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """

    from sample_workchains import Norm
    from aiida_optimize.engines import NelderMead
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.work.launch import run
    tolerance = 1e-1
    result = run(
        WorkflowFactory('optimize.optimize'),
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


def test_nelder_mead_rosenbrock(configure):  # pylint: disable=unused-argument
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """

    from sample_workchains import RosenbrockFunction
    from aiida_optimize.engines import NelderMead
    from aiida.orm import WorkflowFactory, load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.work.launch import run
    tolerance = 1e-1
    result = run(
        WorkflowFactory('optimize.optimize'),
        engine=NelderMead,
        engine_kwargs=ParameterData(
            dict=dict(
                xtol=tolerance,
                ftol=tolerance,
                simplex=[[1., 1.], [1., 2.], [2., 1.]],
                result_key='result'
            )
        ),
        calculation_workchain=RosenbrockFunction,
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
    from aiida.work.launch import submit
    tolerance = 0.1
    pk = submit(  # pylint: disable=invalid-name
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
    ).pk
    wait_for(pk)
    calc = load_node(pk)
    calc_uuid = calc.out.calculation_uuid
    print(calc_uuid)
    uuid_value = calc_uuid.value
    print(uuid_value)
    opt_calc_node = load_node(uuid_value)
    print(opt_calc_node)
    assert np.isclose(opt_calc_node.out.result.value, 0., atol=tolerance)
    assert np.isclose(calc.out.optimizer_result.value, 0, atol=tolerance)
