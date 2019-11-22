# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np
from aiida.engine import workfunction
from aiida.orm import Float


@workfunction
def sin(x):
    # This is a bit improper: The new value should be created in a calculation.
    return Float(np.sin(x.value)).store()
