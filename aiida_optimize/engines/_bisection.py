"""
Defines a 1D bisection optimization engine.
"""

from __future__ import division, print_function, unicode_literals

from fsc.export import export

from aiida.orm.data.base import Float

from ._base import OptimizationEngine


@export
class Bisection(OptimizationEngine):
    """
    TODO
    """

    def __init__(self, lower, upper, tol=1e-6, result_state=None):
        super(Bisection, self).__init__(result_state=result_state)
        self.lower, self.upper = sorted([lower, upper])
        self.tol = tol

    @property
    def _state(self):
        return {k: v for k, v in self.__dict__.items() if k not in ['_result_mapping']}

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
        res = outputs.values()[0]['result']
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
