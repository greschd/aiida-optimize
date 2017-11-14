"""
Defines the WorkChain which runs the optimization procedure.
"""

from contextlib import contextmanager

from fsc.export import export
from aiida_tools import check_workchain_step
from aiida_tools.workchain_inputs import WORKCHAIN_INPUT_KWARGS

from aiida.orm.data.parameter import ParameterData
from aiida.work.workchain import WorkChain, ToContext, while_
from aiida.work import submit


@export  # pylint: disable=abstract-method
class OptimizationWorkChain(WorkChain):
    """
    Runs an optimization procedure, given an optimization engine that defines the optimization algorithm, and a CalculationWorkChain which evaluates the function to be optimized.
    """
    _CALC_PREFIX = 'calc_'

    @classmethod
    def define(cls, spec):
        super(cls, OptimizationWorkChain).define(spec)

        spec.input(
            'engine',
            help='Engine that runs the optimization.',
            **WORKCHAIN_INPUT_KWARGS
        )
        spec.input(
            'engine_kwargs',
            valid_type=ParameterData,
            help='Keyword arguments passed to the optimization engine.'
        )
        spec.input(
            'calculation_workchain',
            help='WorkChain which produces the result to be optimized.',
            **WORKCHAIN_INPUT_KWARGS
        )

        spec.outline(
            cls.create_optimizer,
            while_(cls.not_finished)(cls.launch_calculations, cls.get_results),
            cls.finalize
        )
        spec.output('optimizer_result')

    @contextmanager
    def optimizer(self):
        optimizer = self.engine.from_state(self.ctx.optimizer_state)
        yield optimizer
        self.ctx.optimizer_state = optimizer.state

    @property
    def engine(self):
        return self.get_deserialized_input('engine')

    @property
    def indices_to_retrieve(self):
        return self.ctx.setdefault('indices_to_retrieve', [])

    @indices_to_retrieve.setter
    def indices_to_retrieve(self, value):
        self.ctx['indices_to_retrieve'] = value

    @check_workchain_step
    def create_optimizer(self):
        optimizer = self.engine(**self.inputs.engine_kwargs.get_dict())  # pylint: disable=not-callable
        self.ctx.optimizer_state = optimizer.state

    @check_workchain_step
    def not_finished(self):
        """
        Check if the optimization needs to continue.
        """
        with self.optimizer() as opt:
            return not opt.is_finished

    @check_workchain_step
    def launch_calculations(self):
        """
        Create calculations for the current iteration step.
        """
        self.report('Launching pending calculations.')
        with self.optimizer() as opt:
            calcs = {}
            for idx, inputs in opt.create_inputs().items():
                calcs[self.calc_key(idx)] = submit(
                    self.get_deserialized_input('calculation_workchain'),
                    **inputs
                )
                self.report('Launching calculation {}'.format(idx))
                self.indices_to_retrieve.append(idx)
        return ToContext(**calcs)

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
            outputs[idx] = self.ctx[key].get_outputs_dict()

        with self.optimizer() as opt:
            opt.update(outputs)

    @check_workchain_step
    def finalize(self):
        """
        Return the output after the optimization procedure has finished.
        """
        self.report('Finalizing optimization procedure.')
        with self.optimizer() as opt:
            self.out('optimizer_result', opt.result_value)
            result_index = opt.result_index
            # TODO: move this to an output group (namespace) once those are implemented
            result_calculation = self.ctx[self.calc_key(result_index)]
            for label, output in result_calculation.get_outputs(
                also_labels=True
            ):
                self.out('calculation_{}'.format(label), output)

    def calc_key(self, index):
        """
        Returns the calculation key corresponding to a given index.
        """
        return self._CALC_PREFIX + str(index)
