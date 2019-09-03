# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines simple workchains which are used in the tests.
"""

import numpy as np
import scipy.linalg as la

from aiida.orm import Float, List
from aiida.engine import WorkChain, workfunction

from aiida_tools import check_workchain_step


class Echo(WorkChain):
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


class Norm(WorkChain):
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
        res = Float(la.norm(self.inputs.x.get_attr('list')))
        res.store()
        self.out('result', res)


@workfunction
def sin_list(x):
    return Float(np.sin(list(x)[0]))


@workfunction
def rosenbrock(x):
    x, y = x
    return Float((1 - x)**2 + 100 * (y - x**2)**2)


@workfunction
def add(x, y):
    res = Float(x + y)
    res.store()
    return res
