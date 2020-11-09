# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines a parameter sweep optimization engine.
"""

from aiida.orm.nodes.data.base import to_aiida_type

from ..helpers import get_nested_result
from .base import OptimizationEngineImpl, OptimizationEngineWrapper

__all__ = ['ParameterSweep']


class _ParameterSweepImpl(OptimizationEngineImpl):
    """
    Implementation class for the parameter sweep engine.
    """
    def __init__(self, parameters, result_key, logger, result_state=None):
        super(_ParameterSweepImpl, self).__init__(logger=logger, result_state=result_state)
        self._parameters = parameters
        self._result_key = result_key

    @property
    def _state(self):
        return {'parameters': self._parameters, 'result_key': self._result_key}

    @property
    def is_finished(self):
        if len(self._result_mapping) < len(self._parameters):
            return False
        return not any(res.output is None for res in self._result_mapping.values())

    def _create_inputs(self):
        return [{k: to_aiida_type(v)
                 for k, v in param_dict.items()} for param_dict in self._parameters]

    def _update(self, outputs):
        pass

    def _get_optimal_result(self):
        """
        Return the index and optimizatin value of the best evaluation process.
        """
        cost_values = {
            k: get_nested_result(v.output, self._result_key)
            for k, v in self._result_mapping.items()
        }
        opt_index, opt_output = min(cost_values.items(), key=lambda item: item[1].value)
        input_keys = list(self._parameters[opt_index].keys())
        opt_input = self._result_mapping[opt_index].input[input_keys[0]]
        return (opt_index, opt_input, opt_output)


class ParameterSweep(OptimizationEngineWrapper):
    """
    Optimization engine that performs a parameter sweep.

    :param parameters: List of parameter dictionaries. For each entry, an evaluation with the given parameters will be run.
    :type parameters: list(dict)

    :param result_key: Name of the evaluation process output argument.
    :type result_key: str
    """
    _IMPL_CLASS = _ParameterSweepImpl

    def __new__(cls, parameters, result_key='result', logger=None):  # pylint: disable=arguments-differ
        return cls._IMPL_CLASS(parameters=parameters, result_key=result_key, logger=logger)  # pylint: disable=no-member
