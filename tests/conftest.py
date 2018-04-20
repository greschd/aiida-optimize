"""
Configuration file for pytest tests of aiida-optimize.
"""

import os
os.environ['PYTHONPATH'] = (
    os.environ.get('PYTHONPATH', '') + ':' + os.path.dirname(os.path.abspath(__file__))
)

import pytest

from aiida_pytest import *  # pylint: disable=unused-wildcard-import


@pytest.fixture(params=['run', 'submit'])
def run_optimization(request, configure_with_daemon, wait_for):  # pylint: disable=unused-argument,redefined-outer-name
    """
    Checks an optimization engine with the given parameters.
    """

    def inner(  # pylint: disable=missing-docstring
        engine,
        func_workchain,
        engine_kwargs,
    ):
        from aiida_optimize.workchain import OptimizationWorkChain
        from aiida.orm import load_node
        from aiida.orm.data.parameter import ParameterData
        from aiida.work.launch import run, submit
        from aiida.common.links import LinkType

        inputs = dict(
            engine=engine,
            engine_kwargs=ParameterData(dict=dict(engine_kwargs)),
            calculation_workchain=func_workchain
        )

        if request.param == 'run':
            result = run(OptimizationWorkChain, **inputs)
        else:
            assert request.param == 'submit'
            pk = submit(OptimizationWorkChain, **inputs).pk
            wait_for(pk)
            result = load_node(pk).get_outputs_dict(link_type=LinkType.CREATE)
        return result

    return inner
