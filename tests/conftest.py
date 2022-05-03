# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Configuration file for pytest tests of aiida-optimize.
"""

import operator
import os
from collections import ChainMap
import time

import numpy as np
import pytest
from aiida import orm
from aiida.common import exceptions
from aiida.engine.launch import run_get_node, submit
from aiida_optimize import OptimizationWorkChain
from aiida import get_profile

# This is needed so that the daemon can also load the local modules.
os.environ['PYTHONPATH'] = (
    os.environ.get('PYTHONPATH', '') + ':' + os.path.dirname(os.path.abspath(__file__))
)
import sample_processes

pytest_plugins = ['aiida.manage.tests.pytest_fixtures']

@pytest.fixture
def wait_for():
    def inner(pid, timeout=1):
        from aiida.orm import load_node
        calc = load_node(pid)
        while not calc.is_finished:
            time.sleep(timeout)
    return inner

@pytest.fixture
def with_daemon():
    """Starts the daemon process and then makes sure to kill it once the test is done."""
    import subprocess
    import sys

    from aiida.cmdline.utils.common import get_env_with_venv_bin
    from aiida.engine.daemon.client import DaemonClient

    # Add the current python path to the environment that will be used for the daemon sub process.
    # This is necessary to guarantee the daemon can also import all the classes that are defined
    # in this `tests` module.
    env = get_env_with_venv_bin()
    env['PYTHONPATH'] = ':'.join(sys.path)

    profile = get_profile()

    process = subprocess.Popen(
        DaemonClient(profile).cmd_string.split(),
        stderr=sys.stderr,
        stdout=sys.stdout,
        env=env,
    )

    yield
    
    process.terminate()
    


@pytest.mark.usefixtures('aiida_profile_clean')
@pytest.fixture(params=[
    'run', 
    # 'submit', # FIXME: i shall pass
])
def run_optimization(request, with_daemon, wait_for):  # pylint: disable=unused-argument
    """
    Checks an optimization engine with the given parameters.
    """
    def inner(engine, func_workchain, engine_kwargs, evaluate=None):  # pylint: disable=missing-docstring,useless-suppression
        inputs = dict(
            engine=engine,
            engine_kwargs=orm.Dict(dict=dict(engine_kwargs)),
            evaluate_process=func_workchain,
            evaluate=evaluate if evaluate is not None else {},
        )

        if request.param == 'run':
            _, result_node = run_get_node(OptimizationWorkChain, **inputs)
        else:
            assert request.param == 'submit'

            pk = submit(OptimizationWorkChain, **inputs).pk
            wait_for(pk)
            result_node = orm.load_node(pk)
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
        input_getter=operator.attrgetter('x'),
        output_port_names=None
    ):

        func_workchain = getattr(sample_processes, func_workchain_name)

        result_node = run_optimization(
            engine=engine,
            engine_kwargs=ChainMap(engine_kwargs, {'result_key': 'result'}),
            func_workchain=func_workchain,
            evaluate=evaluate
        )

        assert 'optimal_process_uuid' in result_node.outputs
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
        output_port_names=None
    ):

        func_workchain = getattr(sample_processes, func_workchain_name)

        result_node = run_optimization(
            engine=engine,
            engine_kwargs=ChainMap(engine_kwargs, {'result_key': 'result'}),
            func_workchain=func_workchain,
            evaluate=evaluate
        )

        assert result_node.exit_status == exit_status

        if output_port_names is not None:
            for name in output_port_names:
                assert name in result_node.outputs

    return inner
