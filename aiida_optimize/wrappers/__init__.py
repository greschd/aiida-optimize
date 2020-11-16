# -*- coding: utf-8 -*-
"""
Defines helper workchains that can wrap existing processes into
a format compatible with the optimization procedure.
"""

from ._create_evaluate import CreateEvaluateWorkChain
from ._add_inputs import AddInputsWorkChain

from . import _run_or_submit

__all__ = ['CreateEvaluateWorkChain', 'AddInputsWorkChain']
