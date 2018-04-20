"""
Defines simple workchains which are used in the tests.
"""

import scipy.linalg as la

from aiida.orm.data.base import Float, List
from aiida.work.workchain import WorkChain
from aiida.work.workfunctions import workfunction

from aiida_tools import check_workchain_step


class Echo(WorkChain):
    """
    WorkChain which returns the input.
    """

    @classmethod
    def define(cls, spec):
        super(Echo, cls).define(spec)

        spec.input('x', valid_type=Float)
        spec.output('return', valid_type=Float)
        spec.outline(cls.echo)

    @check_workchain_step
    def echo(self):
        self.report('Starting echo')
        self.out('return', self.inputs.x)


class Norm(WorkChain):
    """
    WorkChain which returns the norm of the input list.
    """

    @classmethod
    def define(cls, spec):
        super(Norm, cls).define(spec)

        spec.input('x', valid_type=List)
        spec.output('return', valid_type=Float)
        spec.outline(cls.evaluate)

    @check_workchain_step
    def evaluate(self):
        self.report('Starting evaluate')
        self.out('return', Float(la.norm(self.inputs.x.get_attr('list'))))


# class HimmelblauFunction(WorkChain):
#     """
#     Workchain which evaluates Himmelblau's function.
#     """
#
#     @classmethod
#     def define(cls, spec):
#         super(HimmelblauFunction, cls).define(spec)
#
#         spec.input('x', valid_type=List)
#         spec.output('return', valid_type=Float)
#         spec.outline(cls.evaluate)
#
#     @check_workchain_step
#     def evaluate(self):
#         self.report('Starting evaluate')
#         x, y = self.inputs.x.get_attr('list')
#         res = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
#         self.out('return', Float(res))


@workfunction
def rosenbrock(x):
    x, y = x
    return Float((1 - x)**2 + 100 * (y - x**2)**2)


@workfunction
def add(x, y):
    return Float(x + y)


# class Add(WorkChain):
#     """
#     WorkChain which adds together two values.
#     """
#
#     @classmethod
#     def define(cls, spec):
#         super(Add, cls).define(spec)
#
#         spec.input('x', valid_type=Float)
#         spec.input('y', valid_type=Float)
#         spec.output('return', valid_type=Float)
#         spec.outline(cls.add)
#
#     @check_workchain_step
#     def add(self):
#         self.report('Starting add')
#         self.out('return', Float(self.inputs.x.value + self.inputs.y.value))
