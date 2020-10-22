# -*- coding: utf-8 -*-

# Author: Austin Zadoks <austin.zadoks@epfl.ch>
"""
Tests for the OptimizationWorkChain.
"""

import pytest

from aiida_optimize.engines import Convergence


@pytest.fixture
def convergence_parameters():
    return {
        'input_values': [0, 1, 2, 3, 4, 4, 5, 4, 4, 4],
        'tol': 1e-8,
        'input_key': 'x',
        'result_key': 'result',
        'convergence_window': 3
    }

@pytest.fixture
def convergence_parameters_202():
    return {
        'input_values': [0, 1, 2, 3, 4, 4, 5, 4, 4, 4],
        'tol': 1e-8,
        'input_key': 'x',
        'result_key': 'result',
        'convergence_window': 5
    }

def test_convergence_echo_wf(check_optimization, convergence_parameters):  # pylint: disable=redefined-outer-name
    """
    Simple test of the OptimizationWorkChain with the Convergence engine.
    """

    check_optimization(
        engine=Convergence,
        engine_kwargs=convergence_parameters,
        func_workchain_name='echo_workfunction',
        xtol=0,
        ftol=0,
        x_exact=4,
        f_exact=4
    )

def test_convergence_echo_wc(check_optimization, convergence_parameters):  # pylint: disable=redefined-outer-name
    """
    Test the 202 is_finished_ok failure state of the OptimzationWorkChain with the Convergence engine.
    """

    check_optimization(
        engine=Convergence,
        engine_kwargs=convergence_parameters,
        func_workchain_name='Echo',
        xtol=0,
        ftol=0,
        x_exact=4,
        f_exact=4
    )

def test_convergence_echo_wc_202(check_optimization, convergence_parameters):  # pylint: disable=redefined-outer-name
    """
    Test the 202 is_finished_ok failure state of the OptimzationWorkChain with the Convergence engine.
    """

    check_optimization(
        engine=Convergence,
        engine_kwargs=convergence_parameters,
        func_workchain_name='echo_workfunction',
        xtol=0,
        ftol=0,
        x_exact=4,
        f_exact=4
    )
