# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Configuration file for pytest tests of aiida-optimize.
"""

import os

import numpy as np
import pytest
from aiida_pytest import *  # pylint: disable=unused-wildcard-import,redefined-builtin

os.environ['PYTHONPATH'] = (
    os.environ.get('PYTHONPATH', '') + ':' + os.path.dirname(os.path.abspath(__file__))
)
try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap


@pytest.fixture(params=['run', 'submit'])
def run_optimization(request, configure_with_daemon, wait_for):  # pylint: disable=unused-argument,redefined-outer-name
    """
    Checks an optimization engine with the given parameters.
    """
    def inner(engine, func_workchain, engine_kwargs, calculation_inputs=None):  # pylint: disable=missing-docstring,useless-suppression
        from aiida_optimize.workchain import OptimizationWorkChain
        from aiida.orm import load_node
        from aiida.orm import Dict
        from aiida.engine.launch import run_get_node, submit

        inputs = dict(
            engine=engine,
            engine_kwargs=Dict(dict=dict(engine_kwargs)),
            calculation_workchain=func_workchain,
            calculation_inputs=calculation_inputs if calculation_inputs is not None else {},
        )

        if request.param == 'run':
            _, result_node = run_get_node(OptimizationWorkChain, **inputs)
        else:
            assert request.param == 'submit'
            pk = submit(OptimizationWorkChain, **inputs).pk
            wait_for(pk)
            result_node = load_node(pk)
        return result_node

    return inner


@pytest.fixture
def check_optimization(
    configure,  # pylint: disable=unused-argument,redefined-outer-name
    run_optimization,  # pylint: disable=redefined-outer-name
):
    """
    Runs and checks an optimization with a given engine and parameters.
    """

    def inner(  # pylint: disable=too-many-arguments,missing-docstring,useless-suppression
        engine,
        func_workchain_name,
        engine_kwargs,
        xtol,
        ftol,
        x_exact,
        f_exact,
        calculation_inputs=None,
        input_key='x',
    ):

        from aiida.orm import load_node

        import sample_workchains
        func_workchain = getattr(sample_workchains, func_workchain_name)

        result_node = run_optimization(
            engine=engine,
            engine_kwargs=ChainMap(engine_kwargs, {'result_key': 'result'}),
            func_workchain=func_workchain,
            calculation_inputs=calculation_inputs
        )

        assert 'calculation_uuid' in result_node.outputs
        assert np.isclose(result_node.outputs.optimizer_result.value, f_exact, atol=ftol)
        calc = load_node(result_node.outputs.calculation_uuid.value)
        assert np.allclose(type(x_exact)(getattr(calc.inputs, input_key)), x_exact, atol=xtol)

    return inner
