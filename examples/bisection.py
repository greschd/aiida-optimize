#!/usr/bin/env runaiida

from __future__ import print_function

from aiida.orm.data.parameter import ParameterData
from aiida.work.launch import run

from aiida_optimize.engines import Bisection
from aiida_optimize.workchain import OptimizationWorkChain

from function_workchains import Sin

result = run(
    OptimizationWorkChain,
    engine=Bisection,
    engine_kwargs=ParameterData(dict=dict(upper=1.3, lower=-1., tol=1e-3)),
    calculation_workchain=Sin
)

print('\nResult:', result)
