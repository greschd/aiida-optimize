# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the WorkChain which runs the optimization procedure.
"""

from contextlib import contextmanager
try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap

from fsc.export import export
from aiida_tools import check_workchain_step
from aiida_tools.workchain_inputs import WORKCHAIN_INPUT_KWARGS, load_object

from aiida.orm import Str
from aiida.orm import Dict
from aiida.engine import WorkChain, while_
from aiida.engine.utils import is_process_function
from aiida.engine.launch import run_get_node
from aiida.common.links import LinkType


def _get_outputs_dict(workchain):
    """
    Helper function to mimic the behaviour of the old AiiDA .get_outputs_dict() method.
    """
    if not workchain.is_finished_ok:
        raise ValueError(
            'Optimization failed due to sub-workchain {} not finishing ok.'.format(workchain.pk)
        )
    return {
        link_triplet.link_label: link_triplet.node
        for link_triplet in workchain.get_outgoing(link_type=LinkType.RETURN)
    }


@export
class OptimizationWorkChain(WorkChain):
    """
    Runs an optimization procedure, given an optimization engine that defines the optimization algorithm, and a CalculationWorkChain which evaluates the function to be optimized.
    """
    _CALC_PREFIX = 'calc_'

    @classmethod
    def define(cls, spec):
        super(cls, OptimizationWorkChain).define(spec)

        spec.input('engine', help='Engine that runs the optimization.', **WORKCHAIN_INPUT_KWARGS)
        spec.input(
            'engine_kwargs',
            valid_type=Dict,
            help='Keyword arguments passed to the optimization engine.'
        )
        spec.input(
            'calculation_workchain',
            help='WorkChain which produces the result to be optimized.',
            **WORKCHAIN_INPUT_KWARGS
        )
        spec.input_namespace(
            'calculation_inputs',
            required=False,
            help='Inputs that are passed to all calculation workchains.',
            dynamic=True
        )

        spec.outline(
            cls.create_optimizer,
            while_(cls.not_finished)(cls.launch_calculations, cls.get_results), cls.finalize
        )
        spec.output('optimizer_result')
        spec.output('calculation_uuid')

    @contextmanager
    def optimizer(self):
        optimizer = self.engine.from_state(state=self.ctx.optimizer_state, logger=self)
        yield optimizer
        self.ctx.optimizer_state = optimizer.state

    @property
    def engine(self):
        return load_object(self.inputs.engine.value)

    @property
    def indices_to_retrieve(self):
        return self.ctx.setdefault('indices_to_retrieve', [])

    @indices_to_retrieve.setter
    def indices_to_retrieve(self, value):
        self.ctx.indices_to_retrieve = value

    @check_workchain_step
    def create_optimizer(self):
        self.report('Creating optimizer instance.')
        optimizer = self.engine(logger=self, **self.inputs.engine_kwargs.get_dict())  # pylint: disable=not-callable
        self.ctx.optimizer_state = optimizer.state

    @check_workchain_step
    def not_finished(self):
        """
        Check if the optimization needs to continue.
        """
        self.report('Checking if optimization is finished.')
        with self.optimizer() as opt:
            return not opt.is_finished

    # @check_workchain_step
    def launch_calculations(self):
        """
        Create calculations for the current iteration step.
        """
        self.report('Launching pending calculations.')
        with self.optimizer() as opt:
            calcs = {}
            calculation_workchain = load_object(self.inputs.calculation_workchain.value)
            self.report(calculation_workchain)
            for idx, inputs in opt.create_inputs().items():
                self.report('Launching calculation {}'.format(idx))
                inputs_merged = ChainMap(inputs, self.inputs.get('calculation_inputs', {}))
                if is_process_function(calculation_workchain):
                    _, node = run_get_node(calculation_workchain, **inputs_merged)
                else:
                    node = self.submit(calculation_workchain, **inputs_merged)
                calcs[self.calc_key(idx)] = node
                self.indices_to_retrieve.append(idx)
        return self.to_context(**calcs)

    @check_workchain_step
    def get_results(self):
        """
        Retrieve results of the current iteration step's calculations.
        """
        self.report('Checking finished calculations.')
        outputs = {}
        while self.indices_to_retrieve:
            idx = self.indices_to_retrieve.pop(0)
            key = self.calc_key(idx)
            self.report('Retrieving output for calculation {}'.format(idx))
            outputs[idx] = _get_outputs_dict(self.ctx[key])

        with self.optimizer() as opt:
            opt.update(outputs)

    @check_workchain_step
    def finalize(self):
        """
        Return the output after the optimization procedure has finished.
        """
        self.report('Finalizing optimization procedure.')
        with self.optimizer() as opt:
            optimizer_result = opt.result_value
            optimizer_result.store()
            self.out('optimizer_result', optimizer_result)
            result_index = opt.result_index
            result_calculation = self.ctx[self.calc_key(result_index)]
            calc_uuid = Str(result_calculation.uuid)
            calc_uuid.store()
            self.out('calculation_uuid', calc_uuid)

    def calc_key(self, index):
        """
        Returns the calculation key corresponding to a given index.
        """
        return self._CALC_PREFIX + str(index)
