#!/usr/bin/env runaiida

from __future__ import print_function

from aiida.orm.data.parameter import ParameterData
from aiida.work.launch import run

from aiida_optimize.engines import Bisection
from aiida_optimize.workchain import OptimizationWorkChain

from sin_wf import sin
from sin_wc import Sin

result_wf = run(
    OptimizationWorkChain,
    engine=Bisection,
    engine_kwargs=ParameterData(dict=dict(upper=1.3, lower=-1., tol=1e-3, result_key='result')),
    calculation_workchain=sin
)

result_wc = run(
    OptimizationWorkChain,
    engine=Bisection,
    engine_kwargs=ParameterData(dict=dict(upper=1.3, lower=-1., tol=1e-3, result_key='result')),
    calculation_workchain=Sin
)

print('\nResult with workfunction:', result_wf)
print('\nResult with workchain:', result_wc)
