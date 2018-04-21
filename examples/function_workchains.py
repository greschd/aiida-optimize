import numpy as np

from aiida.orm.data.float import Float
from aiida.work.workchain import WorkChain
from aiida.work.workfunctions import workfunction


@workfunction
def sin(x):
    return Float(np.sin(x.value))


class Sin(WorkChain):
    """
    A simple workchain which represents the function to be optimized.
    """

    @classmethod
    def define(cls, spec):
        super(Sin, cls).define(spec)

        spec.input('x', valid_type=Float)
        spec.output('return', valid_type=Float)

        spec.outline(cls.evaluate)

    def evaluate(self):
        self.out('return', Float(np.sin(self.inputs.x.value)))
