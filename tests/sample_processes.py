# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines simple processes which are used in the tests.
"""

import copy

import numpy as np
import scipy.linalg as la

import pytest

from aiida import orm
from aiida.engine import WorkChain, workfunction, calcfunction
from aiida_tools import check_workchain_step


class Echo(WorkChain):
    """
    WorkChain which returns the input.
    """
    @classmethod
    def define(cls, spec):
        super(Echo, cls).define(spec)

        spec.input('x', valid_type=orm.Float)
        spec.output('result', valid_type=orm.Float)
        spec.outline(cls.echo)

    @check_workchain_step
    def echo(self):
        self.report('Starting echo')
        self.out('result', self.inputs.x)


@workfunction
def echo_workfunction(x):
    return x


@calcfunction
def echo_calcfunction(x):
    return {'result': copy.deepcopy(x)}


@pytest.fixture(params=[Echo, echo_workfunction, echo_calcfunction])
def echo_process(request):
    return request.param


class EchoDictValue(WorkChain):
    """
    WorkChain which echoes a regular value, and a value given in
    a nested Dict.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('x', valid_type=orm.Float)
        spec.input('a', valid_type=orm.Dict)

        spec.output('x', valid_type=orm.Float)
        spec.output('c', valid_type=orm.Float)

        spec.outline(cls.do_echo)

    def do_echo(self):
        self.out('x', self.inputs.x)
        self.out('c', orm.Float(self.inputs.a.get_dict()['b']['c']).store())


class EchoNestedValues(WorkChain):
    """
    Workchain which echoes values in nested namespaces, once as a regular
    input and once in a nested dict.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('x.y', valid_type=orm.Float)
        spec.input('a.b.c.d', valid_type=orm.Dict)

        spec.output('y', valid_type=orm.Float)
        spec.output('f', valid_type=orm.Float)

        spec.outline(cls.do_echo)

    def do_echo(self):
        self.out('y', self.inputs.x.y)
        self.out('f', orm.Float(self.inputs.a.b.c.d.get_dict()['e']['f']).store())


class EchoDifferentNames(WorkChain):
    """
    WorkChain which returns the input, with "non-standard" input / output names.
    """
    @classmethod
    def define(cls, spec):
        super(EchoDifferentNames, cls).define(spec)

        spec.input('y', valid_type=orm.Float)
        spec.output('the_result', valid_type=orm.Float)
        spec.outline(cls.echo)

    @check_workchain_step
    def echo(self):
        self.report('Starting echo')
        self.out('the_result', self.inputs.y)


class Negative(WorkChain):
    """
    WorkChain which returns the negative of the input.
    """
    @classmethod
    def define(cls, spec):
        super(Negative, cls).define(spec)

        spec.input('x', valid_type=orm.Float)
        spec.output('result', valid_type=orm.Float)
        spec.outline(cls.run_negative)

    @check_workchain_step
    def run_negative(self):
        self.report('Starting negative, input {}'.format(self.inputs.x.value))
        self.out('result', orm.Float(-self.inputs.x.value).store())


class Norm(WorkChain):
    """
    WorkChain which returns the norm of the input list.
    """
    @classmethod
    def define(cls, spec):
        super(Norm, cls).define(spec)

        spec.input('x', valid_type=orm.List)
        spec.output('result', valid_type=orm.Float)
        spec.outline(cls.evaluate)

    @check_workchain_step
    def evaluate(self):  # pylint: disable=missing-docstring
        self.report('Starting evaluate')
        res = orm.Float(la.norm(self.inputs.x.get_attribute('list')))
        res.store()
        self.out('result', res)


@workfunction
def sin_list(x):
    return orm.Float(np.sin(list(x)[0])).store()


@workfunction
def rosenbrock(x):
    x, y = x
    return orm.Float((1 - x)**2 + 100 * (y - x**2)**2).store()


@workfunction
def add(x, y):
    return orm.Float(x + y).store()
