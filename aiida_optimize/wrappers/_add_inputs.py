# -*- coding: utf-8 -*-
"""
Defines a workchain for changing the input interface of a process.
"""

from functools import reduce

from aiida_tools.process_inputs import PROCESS_INPUT_KWARGS, load_object

from aiida import orm
from aiida.engine import ToContext
from aiida.common.exceptions import InputValidationError, NotExistent
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.common.extendeddicts import AttributeDict

from .._utils import get_outputs_dict
from ._run_or_submit import RunOrSubmitWorkChain


class AddInputsWorkChain(RunOrSubmitWorkChain):
    """
    Wrapper workchain that takes inputs as keys and values and passes it
    on to a sub-process. This enables taking a process which was not
    designed to be used in optimization, and optimize with respect to
    some arbitrary input. Inputs which always remain the same can be
    specified in the ``inputs`` namespace, whereas the inputs to be
    optimized are given through the ``added_input_keys`` and
    ``added_input_values`` inputs.

    The outputs of the wrapper workchain are the same as those of
    the wrapped process.

    The "added" inputs can only be BaseType sub-classes, or
    attributes of a Dict. For each input, its port location is given
    in the "added_input_keys" input. For example, ``x.y`` would set
    the ``y`` input in the ``x`` namespace.

    For cases where the input is a Dict attribute, the (possibly nested) attribute name is given after a colon. That means ``x:a.b`` would
    set the ``['a']['b']`` attribute of the ``Dict`` given in the ``x``
    input.

    In cases where only a single input needs to be added, they can be
    specified directly instead of wrapped in a List.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            'added_input_values',
            valid_type=(orm.List, orm.BaseType),
            help="Values of the added inputs to be passed into the sub-process."
        )
        spec.input(
            'added_input_keys',
            valid_type=(orm.List, orm.Str),
            help="Specifies the location of each added input."
        )
        spec.input(
            'sub_process',
            **PROCESS_INPUT_KWARGS,
            help="The class of the process that should be wrapped."
        )
        spec.input_namespace(
            'inputs',
            dynamic=True,
            required=False,
            help="Inputs to be passed on to the sub-process."
        )

        spec.exit_code(
            201,
            'ERROR_SUB_PROCESS_FAILED',
            message='Workchain failed because the sub-process did not finish ok.'
        )

        spec.outputs.dynamic = True

        spec.outline(cls.run_process, cls.finalize)

    def run_process(self):
        """
        Merge the inputs namespace and added inputs, and launch the
        sub-process.
        """
        self.report("Merging inputs for the sub-process.")

        if isinstance(self.inputs.added_input_keys, orm.Str):
            added_input_keys = [self.inputs.added_input_keys.value]
            if not isinstance(self.inputs.added_input_values, orm.BaseType):
                raise InputValidationError(
                    "When 'added_input_keys' is given as 'Str', 'added_input_values'"
                    " must be a 'BaseType' instance."
                )
            added_input_values = [self.inputs.added_input_values.value]
        else:
            added_input_keys = self.inputs.added_input_keys.get_list()
            if not isinstance(self.inputs.added_input_values, orm.List):
                raise InputValidationError(
                    "When 'added_input_keys' is given as 'List', 'added_input_values'"
                    " must also be a 'List'."
                )
            added_input_values = self.inputs.added_input_values.get_list()

        if len(added_input_values) != len(added_input_keys):
            raise InputValidationError(
                "Lengths of 'added_input_values' and 'added_input_keys' do not match."
            )

        # This can be replaced by `getattr(self.inputs, 'inputs', {})`
        # once support for AiiDA < 1.3 is dropped.
        # See aiida-core PR #3985 for the relevant feature.
        try:
            inputs = AttributeDict(self.inputs.inputs)
        except (AttributeError, NotExistent):
            inputs = AttributeDict()

        def _get_or_create_sub_dict(in_dict, name):
            try:
                return in_dict[name]
            except KeyError:
                res = {}
                in_dict[name] = res
                return res

        def _get_or_create_port(in_attr_dict, name):
            try:
                return getattr(in_attr_dict, name)
            except AttributeError:
                res = AttributeDict()
                setattr(in_attr_dict, name, res)
                return res

        for key, value in zip(added_input_keys, added_input_values):
            full_port_path, *full_attr_path = key.split(':')
            *port_path, port_name = full_port_path.split('.')
            namespace = reduce(_get_or_create_port, port_path, inputs)
            if not full_attr_path:
                res_value = to_aiida_type(value)
            else:
                assert len(full_attr_path) == 1
                # Get or create the top-level dictionary.
                try:
                    res_dict = getattr(namespace, port_name).get_dict()
                except AttributeError:
                    res_dict = {}

                *sub_dict_path, attr_name = full_attr_path[0].split('.')
                sub_dict = reduce(_get_or_create_sub_dict, sub_dict_path, res_dict)
                sub_dict[attr_name] = value
                res_value = orm.Dict(dict=res_dict).store()

            setattr(namespace, port_name, res_value)

        self.report("Launching the sub-process.")
        return ToContext(
            sub_process=self.run_or_submit(load_object(self.inputs.sub_process.value), **inputs)
        )

    def finalize(self):  # pylint: disable=inconsistent-return-statements
        """
        Retrieve outputs of the sub-process.
        """
        self.report("Retrieving outputs of the sub-process.")
        sub_process = self.ctx.sub_process
        self.out_many(get_outputs_dict(sub_process))
        if not sub_process.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
