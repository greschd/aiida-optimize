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
            help="A list of pairs (label, process_name). The labels can be any "
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
            help="Defines how inputs are passed between sub-processes. "
            "Each list entry entry has the form `((process_label_a, process_label_b), mapping)`, "
            "and defines outputs of process A to be passed to process B. The `mapping` values are "
            "dictionaries `{'output_name': 'input_name'}` giving the output name (in process A) "
            "and input name (in process B) for each value to pass."
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
        process_labels = [label for label, _ in inputs['process_labels'].get_list()]

        if len(set(process_labels)) < len(process_labels):
            return "The 'process_labels' contains duplicate entries."

        for key in inputs['process_inputs']:
            if key not in process_labels:
                return f"The 'process_inputs' namespace contains a sub-namespace '{key}' that does not match any of the 'process_labels'."

        output_input_mappings = inputs['output_input_mappings'].get_list()

        labels_in_mapping = set()
        for labels, _ in output_input_mappings:
            labels_in_mapping.update(labels)
        invalid_labels = labels_in_mapping - set(process_labels)
        if invalid_labels:
            return f"The process labels '{invalid_labels}' used in 'output_input_mappings' do not exist."

        process_label_idx = {label: idx for idx, label in enumerate(process_labels)}
        for (label_a, label_b), _ in output_input_mappings:
            if process_label_idx[label_a] >= process_label_idx[label_b]:
                return f"Cannot pass outputs of '{label_a}' to '{label_b}' as defined in 'output_input_mappings', because '{label_b}' is executed first."

    def _initialize(self):
        self.ctx.process_idx = 0

    def _not_finished(self):
        return self.ctx.process_idx < len(self.inputs.process_labels)

    def _get_current_process(self):
        label, obj_fullname = self.inputs.process_labels.get_list()[self.ctx.process_idx]
        return label, load_object(obj_fullname)

    def _run_sub_process(self):
        label, sub_process_class = self._get_current_process()
        self.report(f"Starting sub-process with label '{label}'.")
        inputs = dict(self.inputs.process_inputs.get(label, {}))

        for (prev_label,
             curr_label), output_input_mapping in self.inputs.output_input_mappings.get_list():
            if curr_label == label:
                prev_outputs = self.ctx.get(f'process_{prev_label}').outputs

                inputs = _merge_nested_keys({
                    input_name: get_nested_result(prev_outputs, output_name)
                    for output_name, input_name in output_input_mapping.items()
                }, inputs)

        return ToContext({f'process_{label}': self.run_or_submit(sub_process_class, **inputs)})

    def _retrieve_sub_process(self):  # pylint: disable=inconsistent-return-statements
        label, _ = self._get_current_process()
        self.report(f"Retrieving outputs for sub-process with label '{label}'.")

        sub_process = self.ctx.get(f'process_{label}')
        self.out(f'process_outputs.{label}', _get_outputs_dict(sub_process, wrap_nested=True))
        if not sub_process.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED

        self.ctx.process_idx += 1
