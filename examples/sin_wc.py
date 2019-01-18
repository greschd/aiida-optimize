# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np

from aiida.orm.data.float import Float
from aiida.work.workchain import WorkChain


class Sin(WorkChain):
    """
    A simple workchain which represents the function to be optimized.
    """

    @classmethod
    def define(cls, spec):
        super(Sin, cls).define(spec)

        spec.input('x', valid_type=Float)
        spec.output('result', valid_type=Float)

        spec.outline(cls.evaluate)

    def evaluate(self):
        self.out('result', Float(np.sin(self.inputs.x.value)))
