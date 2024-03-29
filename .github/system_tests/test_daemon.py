"""Tests to run with a running daemon."""
import operator
import subprocess
import sys
import time

from aiida import orm
from aiida.common import exceptions
from aiida.engine import launch
from aiida.engine.daemon.client import get_daemon_client
import numpy as np

from aiida_optimize import OptimizationWorkChain
from aiida_optimize.engines import Bisection
import sample_processes

TIMEOUTSECS = 4 * 60  # 4 minutes


def print_daemon_log():
    """Print daemon log."""
    daemon_client = get_daemon_client()
    daemon_log = daemon_client.daemon_log_file

    print(f"Output of 'cat {daemon_log}':")
    try:
        print(
            subprocess.check_output(
                ["cat", f"{daemon_log}"],
                stderr=subprocess.STDOUT,
            )
        )
    except subprocess.CalledProcessError as exception:
        print(f"Note: the command failed, message: {exception}")


def wait_for(proc, time_elapsed=5):
    while not proc.is_terminated:
        time.sleep(time_elapsed)


def check_optimization(
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
):  # pylint: disable=too-many-arguments
    """submit launch and check optimization"""
    func_workchain = getattr(sample_processes, func_workchain_name)

    inputs = dict(
        engine=engine,
        engine_kwargs=orm.Dict(dict=dict(engine_kwargs)),
        evaluate_process=func_workchain,
        evaluate=evaluate if evaluate is not None else {},
    )

    result_node = launch.submit(OptimizationWorkChain, **inputs)

    wait_for(result_node)

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
    assert np.allclose(type(x_exact)(getter_input), type(x_exact)(optimal_process_input), atol=xtol)

    if output_port_names is not None:
        for name in output_port_names:
            assert name in result_node.outputs


def launch_all():
    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(
            lower=-1.1,
            upper=1.0,
            tol=tol,
        ),
        func_workchain_name="Echo",
        xtol=tol,
        ftol=tol,
        x_exact=0.0,
        f_exact=0.0,
    )


def main():
    """Submit through daemon"""
    launch_all()

    print_daemon_log()

    sys.exit(0)


if __name__ == "__main__":
    main()
