# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the OptimizationWorkChain.
"""

from __future__ import print_function


def test_bisect(check_optimization):
    """
    Simple test of the OptimizationWorkChain, with the Bisection engine.
    """

    from aiida_optimize.engines import Bisection

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(
            lower=-1.1,
            upper=1.,
            tol=tol,
        ),
        func_workchain_name='Echo',
        xtol=tol,
        ftol=tol,
        x_exact=0.,
        f_exact=0.,
    )

def test_target_value(check_optimization):
    """
    Test of the Bisection engine with a non-zero target value.
    """

    from aiida_optimize.engines import Bisection

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(
            lower=-1.1,
            upper=1.,
            tol=tol,
            target_value=0.5
        ),
        func_workchain_name='Echo',
        xtol=tol,
        ftol=tol,
        x_exact=0.5,
        f_exact=0.5,
    )

def test_input_output_key(check_optimization):
    """
    Test of the Bisection engine with different input / output keys.
    """

    from aiida_optimize.engines import Bisection

    tol = 1e-1
    check_optimization(
        engine=Bisection,
        engine_kwargs=dict(
            lower=-1.1,
            upper=1.,
            tol=tol,
            input_key='y',
            result_key='the_result'
        ),
        func_workchain_name='EchoDifferentNames',
        xtol=tol,
        ftol=tol,
        x_exact=0.,
        f_exact=0.,
        input_key='y',
    )
