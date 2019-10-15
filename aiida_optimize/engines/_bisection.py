# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines a 1D bisection optimization engine.
"""

from __future__ import division, print_function, unicode_literals

from fsc.export import export

from aiida.orm import Float

from ._base import OptimizationEngineImpl, OptimizationEngineWrapper


class _BisectionImpl(OptimizationEngineImpl):
    """
    Implementation class for the bisection optimization engine.
    """
    def __init__(self, lower, upper, tol, result_key, logger, result_state=None):  # pylint: disable=too-many-arguments
        super(_BisectionImpl, self).__init__(logger=logger, result_state=result_state)
        self.lower, self.upper = sorted([lower, upper])
        self.tol = tol
        self.result_key = result_key

    @property
    def _state(self):
        return {k: v for k, v in self.__dict__.items() if k not in ['_result_mapping', '_logger']}

    @property
    def is_finished(self):
        return abs(self.upper - self.lower) < self.tol

    @property
    def average(self):
        return (self.upper + self.lower) / 2.

    def _create_inputs(self):
        return [{'x': Float(self.average)}]

    def _update(self, outputs):
        assert len(outputs.values()) == 1
        res = next(iter(outputs.values()))[self.result_key]
        if res > 0:
            self.upper = self.average
        else:
            self.lower = self.average

    @property
    def result_value(self):
        return Float(self.average)

    @property
    def result_index(self):
        return max(self._result_mapping.keys())


@export
class Bisection(OptimizationEngineWrapper):
    """
    Optimization engine that performs a bisection.

    :param lower: Lower boundary for the bisection.
    :type lower: float

    :param upper: Upper boundary for the bisection.
    :type upper: float

    :param tol: Tolerance in the input value.
    :type tol: float
    """

    _IMPL_CLASS = _BisectionImpl

    def __new__(cls, lower, upper, tol=1e-6, result_key='result', logger=None):  # pylint: disable=arguments-differ
        return cls._IMPL_CLASS(
            lower=lower, upper=upper, tol=tol, result_key=result_key, logger=logger
        )
