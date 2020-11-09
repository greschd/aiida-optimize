# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines a 1D bisection optimization engine.
"""

import typing as ty

from aiida import orm

from ..helpers import get_nested_result
from ._result_mapping import Result
from .base import OptimizationEngineImpl, OptimizationEngineWrapper

__all__ = ['Bisection']


class _BisectionImpl(OptimizationEngineImpl):
    """
    Implementation class for the bisection optimization engine.
    """
    def __init__(
        self,
        *,
        lower: float,
        upper: float,
        tol: float,
        input_key: str,
        result_key: str,
        target_value: float,
        logger: ty.Any,
        result_state: ty.Optional[ty.Dict[int, Result]] = None,
        initialized: bool = False
    ):
        super(_BisectionImpl, self).__init__(logger=logger, result_state=result_state)
        self.lower = lower
        self.upper = upper
        self.initialized = initialized
        self.tol = tol
        self.input_key = input_key
        self.result_key = result_key
        self.target_value = target_value

    @property
    def _state(self) -> ty.Dict[str, ty.Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ['_result_mapping', '_logger']}

    @property
    def is_finished(self) -> bool:
        return abs(self.upper - self.lower) < self.tol

    @property
    def average(self) -> float:
        return (self.upper + self.lower) / 2.

    def _create_inputs(self) -> ty.List[ty.Dict[str, orm.Float]]:
        if not self.initialized:
            return [{
                self.input_key: orm.Float(self.lower)
            }, {
                self.input_key: orm.Float(self.upper)
            }]
        return [{self.input_key: orm.Float(self.average)}]

    def _update(self, outputs: ty.Dict[int, ty.Any]) -> None:
        output_values = outputs.values()
        num_vals = len(output_values)
        if not self.initialized:
            self.initialized = True
            assert num_vals == 2
            # initial step: change upper such that the _result_ of upper is higher
            results = [get_nested_result(val, self.result_key) for val in output_values]
            lower_val, upper_val = results
            if lower_val > upper_val:
                self.lower, self.upper = self.upper, self.lower
            if min(results) > self.target_value:
                # TODO: add exit code
                raise ValueError(
                    "Target value '{}' is outside range '{}'".format(self.target_value, results)
                )
            if max(results) < self.target_value:
                # TODO: add exit code
                raise ValueError(
                    "Target value '{}' is outside range '{}'".format(self.target_value, results)
                )
        else:
            assert num_vals == 1
            res = get_nested_result(next(iter(output_values)), self.result_key)
            if (res - self.target_value) > 0:
                self.upper = self.average
            else:
                self.lower = self.average

    def _get_optimal_result(self) -> ty.Tuple[int, ty.Any, Result]:
        """
        Return the index and optimization value of the best evaluation workflow.
        """
        output_values = {
            key: get_nested_result(value.output, self.result_key)
            for key, value in self._result_mapping.items()
        }
        opt_index, opt_output = min(
            output_values.items(), key=lambda item: abs(item[1].value - self.target_value)
        )
        opt_input = self._result_mapping[opt_index].input[self.input_key]
        return (opt_index, opt_input, opt_output)


class Bisection(OptimizationEngineWrapper):
    """
    Optimization engine that performs a bisection.

    :param lower: Lower boundary for the bisection.
    :type lower: float

    :param upper: Upper boundary for the bisection.
    :type upper: float

    :param tol: Tolerance in the input value.
    :type tol: float

    :param input_key: Name of the input to be varied in the optimization.
    :type input_key: str

    :param result_key: Name of the output which contains the evaluated function.
    :type result_key: str

    :param target_value: Target value of the function towards which it should be optimized.
    :type target_value: float
    """

    _IMPL_CLASS = _BisectionImpl

    def __new__(  # type: ignore  # pylint: disable=arguments-differ
        cls,
        lower: float,
        upper: float,
        *,
        tol: float = 1e-6,
        input_key: str = 'x',
        result_key: str = 'result',
        target_value: float = 0.,
        logger: ty.Optional[ty.Any] = None
    ) -> _BisectionImpl:
        return cls._IMPL_CLASS(  # pylint: disable=no-member
            lower=lower,
            upper=upper,
            tol=tol,
            input_key=input_key,
            result_key=result_key,
            target_value=target_value,
            logger=logger
        )
