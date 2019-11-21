#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

from __future__ import print_function

from aiida.engine.launch import run
from aiida.orm import Dict
from sin_wc import Sin
from sin_wf import sin

from aiida_optimize.engines import Bisection
from aiida_optimize.workchain import OptimizationWorkChain

result_wf = run(
    OptimizationWorkChain,
    engine=Bisection,
    engine_kwargs=Dict(dict=dict(upper=1.3, lower=-1., tol=1e-3, result_key='result')),
    calculation_workchain=sin
)

result_wc = run(
    OptimizationWorkChain,
    engine=Bisection,
    engine_kwargs=Dict(dict=dict(upper=1.3, lower=-1., tol=1e-3, result_key='result')),
    calculation_workchain=Sin
)

print('\nResult with workfunction:', result_wf)
print('\nResult with workchain:', result_wc)
