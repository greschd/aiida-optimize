# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Fixtures for pytest tests of aiida-optimize.
"""

from collections import ChainMap
import operator

from aiida import orm
from aiida.common import exceptions
from aiida.engine.launch import run_get_node
import numpy as np
import pytest

from aiida_optimize import OptimizationWorkChain
import sample_processes
from sample_processes import Echo, echo_calcfunction, echo_workfunction

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


@pytest.fixture(params=[Echo, echo_workfunction, echo_calcfunction])
def echo_process(request):
    return request.param


@pytest.mark.usefixtures("aiida_profile_clean")
@pytest.fixture
def run_optimization():  # pylint: disable=unused-argument
    """
    Checks an optimization engine with the given parameters.
    """

    def inner(
        engine, func_workchain, engine_kwargs, evaluate=None
    ):  # pylint: disable=missing-docstring,useless-suppression

        inputs = dict(
            engine=engine,
            engine_kwargs=orm.Dict(dict=dict(engine_kwargs)),
            evaluate_process=func_workchain,
            evaluate=evaluate if evaluate is not None else {},
        )

        _, result_node = run_get_node(OptimizationWorkChain, **inputs)

        return result_node

    return inner


@pytest.fixture
def check_optimization(
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
        evaluate=None,
        input_getter=operator.attrgetter("x"),
        output_port_names=None,
    ):

        func_workchain = getattr(sample_processes, func_workchain_name)

        result_node = run_optimization(
            engine=engine,
            engine_kwargs=ChainMap(engine_kwargs, {"result_key": "result"}),
            func_workchain=func_workchain,
            evaluate=evaluate,
        )

        assert "optimal_process_uuid" in result_node.outputs
        assert np.isclose(result_node.outputs.optimal_process_output.value, f_exact, atol=ftol)

        calc = orm.load_node(result_node.outputs.optimal_process_uuid.value)
        assert np.allclose(type(x_exact)(input_getter(calc.inputs)), x_exact, atol=xtol)

        try:
            optimal_process_input_node = result_node.outputs.optimal_process_input
        except exceptions.NotExistentAttributeError:
            return

        if isinstance(optimal_process_input_node, orm.BaseType):
            optimal_process_input = optimal_process_input_node.value
        elif isinstance(optimal_process_input_node, orm.List):
            optimal_process_input = optimal_process_input_node.get_list()
        else:
            optimal_process_input = optimal_process_input_node

        getter_input = input_getter(calc.inputs)
        if isinstance(getter_input, orm.Node):
            assert getter_input.uuid == optimal_process_input_node.uuid

        assert np.allclose(type(x_exact)(optimal_process_input), x_exact, atol=xtol)
        assert np.allclose(
            type(x_exact)(getter_input), type(x_exact)(optimal_process_input), atol=xtol
        )

        if output_port_names is not None:
            for name in output_port_names:
                assert name in result_node.outputs

    return inner


@pytest.fixture
def check_error(
    run_optimization,  # pylint: disable=redefined-outer-name
):
    """
    Runs and checks an optimization with a given engine and parameters.
    """

    def inner(  # pylint: disable=too-many-arguments,missing-docstring,useless-suppression
        engine,
        func_workchain_name,
        engine_kwargs,
        exit_status,
        evaluate=None,
        output_port_names=None,
    ):

        func_workchain = getattr(sample_processes, func_workchain_name)

        result_node = run_optimization(
            engine=engine,
            engine_kwargs=ChainMap(engine_kwargs, {"result_key": "result"}),
            func_workchain=func_workchain,
            evaluate=evaluate,
        )

        assert result_node.exit_status == exit_status

        if output_port_names is not None:
            for name in output_port_names:
                assert name in result_node.outputs

    return inner
