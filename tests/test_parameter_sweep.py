# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the OptimizationWorkChain.
"""

import numpy as np
import pytest

from aiida import orm

from aiida_optimize.engines import ParameterSweep


@pytest.fixture
def sweep_parameters():
    return [{'x': x} for x in np.linspace(-2, 2, 10)]


def test_parameter_sweep(check_optimization, sweep_parameters):  # pylint: disable=redefined-outer-name
    """
    Simple test of the OptimizationWorkChain with the ParameterSweep engine.
    """

    check_optimization(
        engine=ParameterSweep,
        engine_kwargs=dict(parameters=sweep_parameters),
        func_workchain_name='Echo',
        xtol=0,
        ftol=0,
        x_exact=-2.,
        f_exact=-2.,
    )


def test_parameter_sweep_add(check_optimization, sweep_parameters):  # pylint: disable=redefined-outer-name
    """
    Test the ParameterSweep Engine with the add workfunction, using 'evaluate'.
    """

    check_optimization(
        engine=ParameterSweep,
        engine_kwargs=dict(parameters=sweep_parameters),
        func_workchain_name='add',
        xtol=0,
        ftol=0,
        x_exact=-2.,
        f_exact=-1.,
        evaluate={'y': orm.Float(1.)}
    )
