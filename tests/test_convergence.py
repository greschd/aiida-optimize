# -*- coding: utf-8 -*-

# Author: Austin Zadoks <austin.zadoks@epfl.ch>
"""
Tests for the Convergence engine.
"""
from aiida import orm

from aiida_optimize.engines import Convergence


def test_convergence_echo_wf(check_optimization):
    """
    Simple test of the OptimizationWorkChain with the Convergence engine.
    """

    check_optimization(
        engine=Convergence,
        engine_kwargs={
            "input_values": [0, 1, 2, 3, 4.001, 4.002, 5, 4.003, 4.004, 4.005],
            "tol": 1e-1,
            "input_key": "x",
            "result_key": "result",
            "convergence_window": 3,
        },
        func_workchain_name="echo_workfunction",
        xtol=0,
        ftol=0,
        x_exact=4.003,
        f_exact=4.003,
    )


def test_convergence_echo_nested_wf(check_optimization):
    """
    Simple test of the OptimizationWorkChain with the Convergence engine.
    """

    check_optimization(
        engine=Convergence,
        engine_kwargs={
            "input_values": [0, 1, 2, 3, 4.001, 4.002, 5, 4.003, 4.004, 4.005],
            "tol": 1e-1,
            "input_key": "a.b.c.d:e.f",
            "result_key": "f",
            "convergence_window": 3,
        },
        func_workchain_name="EchoNestedValues",
        xtol=0,
        ftol=0,
        x_exact=4.003,
        f_exact=4.003,
        evaluate={"x": {"y": orm.Float(0.0)}},
        input_getter=lambda inputs: inputs.a__b__c__d["e"]["f"],
    )


def test_convergence_echo_wf_202(check_error):
    """
    Test the 202 is_finished_ok failure state of the OptimzationWorkChain with the Convergence engine.
    """

    check_error(
        engine=Convergence,
        engine_kwargs={
            "input_values": [0, 1, 2, 3, 4.001, 4.002, 5, 4.003, 4.004, 4.005],
            "tol": 1e-1,
            "input_key": "x",
            "result_key": "result",
            "convergence_window": 5,
        },
        func_workchain_name="echo_workfunction",
        exit_status=202,
    )
