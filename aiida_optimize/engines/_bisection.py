from fsc.export import export

from aiida.orm import DataFactory
from aiida.orm.data.base import Float

from ._result_mapping import ResultMapping


@export
class Bisection(object):
    """
    TODO
    """

    def __init__(self, lower, upper, tol=1e-6, result_state=None):
        self._result_mapping = ResultMapping.from_state(result_state)
        self.lower, self.upper = sorted([lower, upper])
        self.tol = tol

    @classmethod
    def from_state(cls, state):
        return cls(**state.get_dict())

    @property
    def state(self):
        return DataFactory('parameter')(
            dict=dict(
                result_state=self._result_mapping.state,
                **{
                    k: v
                    for k, v in self.__dict__.items()
                    if k not in ['_result_mapping']
                }
            )
        )

    @property
    def is_finished(self):
        return abs(self.upper - self.lower) < self.tol

    @property
    def average(self):
        return (self.upper + self.lower) / 2.

    def create_inputs(self):
        return self._result_mapping.add_inputs(self._create_inputs())

    def _create_inputs(self):
        return [{'x': Float(self.average)}]

    def update(self, outputs):
        self._result_mapping.add_outputs(outputs)
        res = outputs.values()[0]['result']
        print(res)
        if res > 0:
            self.upper = self.average
        else:
            self.lower = self.average

    @property
    def result(self):
        return Float(self.average)
