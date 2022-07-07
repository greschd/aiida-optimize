# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

from aiida.engine import WorkChain
from aiida.orm.nodes.data.float import Float
import numpy as np


class Sin(WorkChain):
    """
    A simple workchain which represents the function to be optimized.
    """

    @classmethod
    def define(cls, spec):
        super(Sin, cls).define(spec)

        spec.input("x", valid_type=Float)
        spec.output("result", valid_type=Float)

        spec.outline(cls.evaluate)

    def evaluate(self):
        # This is a bit improper: The new value should be created in a calculation.
        self.out("result", Float(np.sin(self.inputs.x.value)).store())
