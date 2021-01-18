"""
Defines a wrapper workchain to combine multiple processes.
"""

from aiida import orm
from aiida.engine import while_, ToContext

from aiida_tools.process_inputs import load_object

from .._utils import _get_outputs_dict, _merge_nested_keys
from ..helpers import get_nested_result
from ._run_or_submit import RunOrSubmitWorkChain

__all__ = ("ConcatenateWorkChain", )


class ConcatenateWorkChain(RunOrSubmitWorkChain):
    """Allows concatenating an arbitrary number of sub-processes.

    A wrapper workchain that allows concatenating an arbitrary number
    of sub-processes. Outputs of one processes can be configured to
    be passed to the next one.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            'process_labels',
            valid_type=orm.List,
            help="A list of pairs (label, process_name). The labels can be any"
            "string, the process_name needs to be loadable by "
            "`aiida_tools.process_inputs.load_object`, and defines which "
            "process is being run.",
        )
        spec.input_namespace(
            'process_inputs',
            dynamic=True,
            help="Inputs which are passed on to the sub-processes. "
            "The inputs should be grouped into a namespace identified "
            "by the process label."
        )

        spec.input(
            'output_input_mappings',
            valid_type=orm.List,
            help="A list of dictionaries. The i-th dictionary defines"
            "which outputs of the i-th process are passed to the i+1th. "
            "Keys of the dictionaries are output names, and values input names. "
            "The length of this input must be one shorter than 'process_labels'."
        )

        spec.inputs.validator = cls._validate_inputs

        spec.output_namespace('process_outputs', dynamic=True)

        spec.exit_code(
            200,
            "ERROR_SUB_PROCESS_FAILED",
            message="Workchain failed because a sub-process failed."
        )

        spec.outline(
            cls._initialize,
            while_(cls._not_finished)(cls._run_sub_process, cls._retrieve_sub_process)
        )

    @classmethod
    def _validate_inputs(cls, inputs, ctx=None):  # pylint: disable=unused-argument,inconsistent-return-statements
        """
        Validate that the 'process_labels', 'output_input_mappings' and 'process_inputs'
        are consistent.
        """
        num_labels = len(inputs['process_labels'])
        num_output_input_mappings = len(inputs['output_input_mappings'])
        if num_output_input_mappings != num_labels - 1:
            return "The 'process_labels' and 'output_input_mappings' inputs have inconsistent length."

        labels = [label for label, _ in inputs['process_labels'].get_list()]
        for key in inputs['process_inputs']:
            if key not in labels:
                return f"The 'process_inputs' namespace contains a sub-namespace '{key}' that does not match any of the 'process_labels'."

    def _initialize(self):
        self.ctx.process_idx = 0
        self.ctx.last_label = None

    def _not_finished(self):
        return self.ctx.process_idx < len(self.inputs.process_labels)

    def _get_current_process(self):
        label, obj_fullname = self.inputs.process_labels.get_list()[self.ctx.process_idx]
        return label, load_object(obj_fullname)

    def _run_sub_process(self):
        label, sub_process_class = self._get_current_process()
        inputs = dict(self.inputs.process_inputs.get(label, {}))

        if self.ctx.process_idx > 0:
            output_input_mapping = self.inputs.output_input_mappings.get_list()[self.ctx.process_idx
                                                                                - 1]
            last_outputs = self.ctx.get(f'process_{self.ctx.last_label}').outputs

            inputs = _merge_nested_keys({
                input_name: get_nested_result(last_outputs, output_name)
                for output_name, input_name in output_input_mapping.items()
            }, inputs)

        return ToContext({f'process_{label}': self.run_or_submit(sub_process_class, **inputs)})

    def _retrieve_sub_process(self):  # pylint: disable=inconsistent-return-statements
        label, _ = self._get_current_process()

        sub_process = self.ctx.get(f'process_{label}')
        self.out(f'process_outputs.{label}', _get_outputs_dict(sub_process))
        if not sub_process.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

        self.ctx.last_label = label
        self.ctx.process_idx += 1
