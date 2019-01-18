# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np

from aiida.orm.data.float import Float
from aiida.work.workfunctions import workfunction


@workfunction
def sin(x):
    return Float(np.sin(x.value))
