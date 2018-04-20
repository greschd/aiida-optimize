"""
Configuration file for pytest tests of aiida-optimize.
"""

import os
os.environ['PYTHONPATH'] = (
    os.environ.get('PYTHONPATH', '') + ':' + os.path.dirname(os.path.abspath(__file__))
)
try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap

import numpy as np
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
            result = load_node(pk).get_outputs_dict()
        return result

    return inner


@pytest.fixture
def check_optimization(
    configure,  # pylint: disable=unused-argument
    run_optimization,
):
    """
    Runs and checks an optimization with a given engine and parameters.
    """

    def inner(
        engine,
        func_workchain_name,
        engine_kwargs,
        xtol,
        ftol,
        x_exact,
        f_exact,
        check_x,
    ):

        from aiida.orm import load_node

        import sample_workchains
        func_workchain = getattr(sample_workchains, func_workchain_name)

        result = run_optimization(
            engine=engine,
            engine_kwargs=ChainMap(engine_kwargs, {
                'result_key': 'return'
            }),
            func_workchain=func_workchain,
        )

        assert 'calculation_uuid' in result
        assert np.isclose(result['optimizer_result'].value, f_exact, atol=ftol)
        calc = load_node(result['calculation_uuid'].value)
        if check_x:
            assert np.allclose(list(calc.get_inputs_dict()['x']), x_exact, atol=xtol)

    return inner
