# -*- coding: utf-8 -*-
"""
Defines a workchain that binds together two existing processes.
"""

from aiida import orm
from aiida.engine import ToContext
from aiida.common.exceptions import NotExistent

from aiida_tools.process_inputs import PROCESS_INPUT_KWARGS, load_object

from .._utils import _get_outputs_dict
from ._run_or_submit import RunOrSubmitWorkChain


class CreateEvaluateWorkChain(RunOrSubmitWorkChain):
    """
    Wrapper workchain to combine two processes: The first process _creates_
    a result, and the second _evaluates_ that result.

    The purpose of this workchain is to facilitate optimization of processes
    which don't natively produce an output that can be optimized, by only
    having to add the 'evaluation' part.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            'create_process',
            **PROCESS_INPUT_KWARGS,
            help="The sub-process which performs the create step."
        )
        spec.input(
            'evaluate_process',
            **PROCESS_INPUT_KWARGS,
            help="The sub-process which performs the evaluate step."
        )

        spec.input_namespace(
            'create',
            dynamic=True,
            required=True,
            help="Inputs which are passed on to the create sub-process."
        )
        spec.input_namespace(
            'evaluate',
            dynamic=True,
            required=False,
            help="Inputs which are passed on to the evaluate sub-process."
        )

        spec.input(
            'output_input_mapping',
            valid_type=orm.Dict,
            help="A mapping from output names of the create process to input "
            "names of the evaluate process. These outputs (if present) "
            "are forwarded to the evaluate process."
        )

        spec.output_namespace('create', dynamic=True)
        spec.output_namespace('evaluate', dynamic=True)

        spec.exit_code(
            201,
            "ERROR_CREATE_PROCESS_FAILED",
            message="Workchain failed because the 'create' sub-process failed."
        )
        spec.exit_code(
            202,
            "ERROR_EVALUATE_PROCESS_FAILED",
            message="Workchain failed because the 'evaluate' sub-process failed."
        )

        spec.outline(cls.run_create, cls.run_evaluate, cls.finalize)

    def run_create(self):
        """
        Launch the first, "create" sub-process.
        """
        self.report(f"Running create process '{self.inputs.create_process.value}'")

        create_process_class = load_object(self.inputs.create_process.value)

        return ToContext(
            create_process=self.run_or_submit(create_process_class, **self.inputs.create)
        )

    def run_evaluate(self):
        """
        Retrieve outputs of the "create" sub-process, and launch the
        "evaluate" sub-process.
        """
        create_process_outputs = _get_outputs_dict(self.ctx.create_process)
        self.out('create', create_process_outputs)
        if not self.ctx.create_process.is_finished_ok:
            return self.exit_codes.ERROR_CREATE_PROCESS_FAILED

        self.report(f"Running evaluate process '{self.inputs.evaluate_process.value}'")
        evaluate_process_class = load_object(self.inputs.evaluate_process.value)

        output_input_mapping = self.inputs.output_input_mapping.get_dict()
        created_inputs = {
            in_key: create_process_outputs[out_key]
            for out_key, in_key in output_input_mapping.items() if out_key in create_process_outputs
        }
        # This can be replaced by `getattr(self.inputs, 'evaluate', {})`
        # once support for AiiDA < 1.3 is dropped.
        # See aiida-core PR #3985 for the relevant feature.
        try:
            evaluate_inputs = self.inputs.evaluate
        except (AttributeError, NotExistent):
            evaluate_inputs = {}
        return ToContext(
            evaluate_process=self.
            run_or_submit(evaluate_process_class, **evaluate_inputs, **created_inputs)
        )

    def finalize(self):  # pylint: disable=inconsistent-return-statements
        """
        Retrieve outputs of the "evaluate" sub-process.
        """
        self.report("Retrieving evaluation outputs.")

        self.out('evaluate', _get_outputs_dict(self.ctx.evaluate_process))
        if not self.ctx.evaluate_process.is_finished_ok:
            return self.exit_codes.ERROR_EVALUATE_PROCESS_FAILED
