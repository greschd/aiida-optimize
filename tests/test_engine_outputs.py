# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the OptimizationWorkChain.
"""

from aiida import orm

from aiida_optimize.engines._bisection import Bisection, _BisectionImpl
from aiida_optimize.engines.base import OptimizationEngineImplWithOutputs
from sample_processes import Echo


class BisectionWithCustomOutputImpl(_BisectionImpl, OptimizationEngineImplWithOutputs):
    def get_engine_outputs(self):
        return {"a": orm.Int(3).store(), "b": orm.Float(2.3).store()}


class BisectionWithCustomOutput(Bisection):
    _IMPL_CLASS = BisectionWithCustomOutputImpl


def test_custom_output(run_optimization):
    """
    Simple test of the OptimizationWorkChain, with the Bisection engine.
    """

    tol = 1e-1
    node = run_optimization(
        engine=BisectionWithCustomOutput,
        engine_kwargs=dict(
            lower=-1.1,
            upper=1.0,
            tol=tol,
        ),
        func_workchain=Echo,
    )
    assert "engine_outputs__a" in node.outputs
    assert "engine_outputs__b" in node.outputs
    assert node.outputs.engine_outputs__a.value == 3
    assert node.outputs.engine_outputs__b.value == 2.3
