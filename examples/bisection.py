#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import sys
from os.path import abspath, dirname

from aiida.engine.launch import run
from aiida.orm import Dict

sys.path.append(dirname(abspath(__file__)))
from sin_wc import Sin
from sin_wf import sin

from aiida_optimize import OptimizationWorkChain
from aiida_optimize.engines import Bisection

result_wf = run(
    OptimizationWorkChain,
    engine=Bisection,
    engine_kwargs=Dict(dict=dict(upper=1.3, lower=-1., tol=1e-3, result_key='result')),
    evaluate_process=sin
)

result_wc = run(
    OptimizationWorkChain,
    engine=Bisection,
    engine_kwargs=Dict(dict=dict(upper=1.3, lower=-1., tol=1e-3, result_key='result')),
    evaluate_process=Sin
)

print('\nResult with workfunction:', result_wf)
print('\nResult with workchain:', result_wc)
