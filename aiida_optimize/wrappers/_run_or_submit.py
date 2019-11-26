# -*- coding: utf-8 -*-
"""
Defines a WorkChain base class that implements a way to either
run or submit a sub-process based on its capabilities.
"""

import typing as ty

from aiida import orm
from aiida.engine import Process, WorkChain, run_get_node, utils


class RunOrSubmitWorkChain(WorkChain):
    """
    Adds a 'run_or_submit' method to the WorkChain class, which uses
    'run' for process functions and 'submit' else.
    """
    def run_or_submit(self, proc: Process, **kwargs: ty.Any) -> orm.ProcessNode:
        if utils.is_process_function(proc):
            _, node = run_get_node(proc, **kwargs)
            return node
        return self.submit(proc, **kwargs)
