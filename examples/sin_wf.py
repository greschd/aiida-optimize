import numpy as np

from aiida.orm.data.float import Float
from aiida.work.workfunctions import workfunction


@workfunction
def sin(x):
    return Float(np.sin(x.value))
