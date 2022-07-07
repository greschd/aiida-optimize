# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the OptimizationWorkChain.
"""

import operator

import pytest

from aiida_optimize.engines import Bisection


def test_bisect(check_optimization):
    """
    Simple test of the OptimizationWorkChain, with the Bisection engine.
    """

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


def test_bisect_switch_lower_upper(check_optimization):
    """
    Test bisection with switched values for upper / lower.
    """

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(
            upper=-1.1,
            lower=1.0,
            tol=tol,
        ),
        func_workchain_name="Echo",
        xtol=tol,
        ftol=tol,
        x_exact=0.0,
        f_exact=0.0,
    )


def test_bisect_negative(check_optimization):
    """
    Simple test of the OptimizationWorkChain, with the Bisection engine.
    """

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(lower=-2.0, upper=1.0, tol=tol, target_value=-0.2),
        func_workchain_name="Negative",
        xtol=tol,
        ftol=tol,
        x_exact=0.2,
        f_exact=-0.2,
    )


def test_target_value(check_optimization):
    """
    Test of the Bisection engine with a non-zero target value.
    """

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(lower=-1.1, upper=1.0, tol=tol, target_value=0.5),
        func_workchain_name="Echo",
        xtol=tol,
        ftol=tol,
        x_exact=0.5,
        f_exact=0.5,
    )


def test_input_output_key(check_optimization):
    """
    Test of the Bisection engine with different input / output keys.
    """

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(lower=-1.1, upper=1.0, tol=tol, input_key="y", result_key="the_result"),
        func_workchain_name="EchoDifferentNames",
        xtol=tol,
        ftol=tol,
        x_exact=0.0,
        f_exact=0.0,
        input_getter=operator.attrgetter("y"),
    )


@pytest.mark.parametrize(
    ["input_key", "result_key", "input_getter"],
    [
        ("x", "x", operator.attrgetter("x")),
        ("a:b.c", "c", lambda inp_: inp_.a["b"]["c"]),
        ("f.g", "d.e:f.g", operator.attrgetter("f__g")),
    ],
)
def test_nested_input_output_keys(check_optimization, input_key, result_key, input_getter):
    """
    Test of the Bisection engine with different input / output keys.
    """

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(
            lower=-1.1, upper=1.0, tol=tol, input_key=input_key, result_key=result_key
        ),
        func_workchain_name="EchoDictValue",
        xtol=tol,
        ftol=tol,
        x_exact=0.0,
        f_exact=0.0,
        input_getter=input_getter,
    )


def test_exact_value(check_optimization):
    """
    Check that the exact value is returned, even if it is not the
    last value evaluated.
    """

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(
            lower=0,
            upper=1.0,
            tol=tol,
        ),
        func_workchain_name="Echo",
        xtol=0.0,
        ftol=0.0,
        x_exact=0.0,
        f_exact=0.0,
    )


def test_multiple_input_keys(check_optimization):
    """
    Pass the same input twice by passing a list as `input_key`.
    """
    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(lower=-1.0, upper=1.0, tol=tol, input_key=["x", "y"]),
        func_workchain_name="add",
        xtol=0.0,
        ftol=0.0,
        x_exact=0.0,
        f_exact=0.0,
    )
