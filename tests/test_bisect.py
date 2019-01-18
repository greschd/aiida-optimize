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
