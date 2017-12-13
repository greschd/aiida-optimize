"""
Defines simple workchains which are used in the tests.
"""

import scipy.linalg as la

from aiida.orm.data.base import Float, List
from aiida.work.workchain import WorkChain

from aiida_tools import check_workchain_step


class Echo(WorkChain):  # pylint: disable=abstract-method
    """
    WorkChain which returns the input.
    """

    @classmethod
    def define(cls, spec):
        super(Echo, cls).define(spec)

        spec.input('x', valid_type=Float)
        spec.output('result', valid_type=Float)
        spec.outline(cls.echo)

    @check_workchain_step
    def echo(self):
        self.report('Starting echo')
        self.out('result', self.inputs.x)


class Norm(WorkChain):  # pylint: disable=abstract-method
    """
    WorkChain which returns the norm of the input list.
    """

    @classmethod
    def define(cls, spec):
        super(Norm, cls).define(spec)

        spec.input('x', valid_type=List)
        spec.output('result', valid_type=Float)
        spec.outline(cls.evaluate)

    @check_workchain_step
    def evaluate(self):
        self.report('Starting evaluate')
        self.out('result', Float(la.norm(self.inputs.x.get_attr('list'))))


class Add(WorkChain):  # pylint: disable=abstract-method
    """
    WorkChain which adds together two values.
    """

    @classmethod
    def define(cls, spec):
        super(Add, cls).define(spec)

        spec.input('x', valid_type=Float)
        spec.input('y', valid_type=Float)
        spec.output('result', valid_type=Float)
        spec.outline(cls.add)

    @check_workchain_step
    def add(self):
        self.report('Starting add')
        self.out('result', Float(self.inputs.x.value + self.inputs.y.value))
