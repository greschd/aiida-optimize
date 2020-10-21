# -*- coding: utf-8 -*-

# Author: Austin Zadoks <austin.zadoks@epfl.ch>
"""
Defines a basic and generic parameter convergence engine.
"""

import itertools
import typing as ty

import numpy as np
from aiida import orm
from aiida_optimize.engines._base import (OptimizationEngineImpl,
                                          OptimizationEngineWrapper)
from aiida_optimize.engines._result_mapping import Result

__all__ = ['Convergence']


class _ConvergenceImpl(OptimizationEngineImpl):
    """
    Implementation class for the convergence engine.
    """
    def __init__(
        self,
        *,
        input_values: ty.Iterable[ty.Any],
        tol: float,
        convergence_window: int,
        input_key: str,
        result_key: str,
        current_index: int,
        result_values: ty.List[ty.Any],
        initialized: bool,
        logger: ty.Optional[ty.Any],
        result_state: ty.Optional[ty.Dict[int, Result]] = None,
    ):
        super(_ConvergenceImpl, self).__init__(logger=logger, result_state=result_state)
        self.input_values = input_values
        self.tol = tol
        self.input_key = input_key
        self.result_key = result_key
        self.convergence_window = convergence_window
        self.current_index = current_index
        self.result_values = result_values
        self.initialized = initialized

    @property
    def _state(self) -> ty.Dict[str, ty.Any]:
        """
        Generate the engine state, including variables
            input_values, tol, input_key, result_key, convergence_window,
            current_index, result_values, initialized
        and excluding variables
            logger
        """
        return {k: v for k, v in self.__dict__.items() if k not in ['_result_mapping', '_logger']}

    @property
    def is_finished(self) -> bool:
        """
        Check if convergence has been reached by calculating the Frobenius
        or 2-norm of the difference between all the result values / arrays
        and checking that the maximum distance between points in the
        convergence window is less than the tolerance.
        """
        if not self.initialized:
            return False

        if self.current_index + 1 >= len(self.input_values):
            return True

        result_window = self.result_values[-self.convergence_window:]
        
        # |x_i - x_j| for i [0, N-1], j (i, N]
        distance_triangle = [[
            np.linalg.norm(result_window[i].value - result_window[j].value)
            for j in range(i + 1, self.convergence_window)
        ] for i in range(self.convergence_window - 1)]
        
        # flatten all the distances into a 1D list
        distances = list(itertools.chain(*distance_triangle))
        
        # check if the maximum distance is less than the tolerance
        return np.max(distances) < self.tol

    @property
    def is_finished_ok(self) -> bool:
        if self.is_finished and self.current_index + 1 >= len(self.input_values):
            return True
        return False

    def _create_inputs(self) -> ty.List[ty.Dict[str, orm.Float]]:
        """
        Create the inputs for the evaluation function.
        If the work chain is not initialized, the appropriate number of
        inputs to fill the convergence window are generated.
        Otherwise, one more input will be generated
        """
        # TODO: should we be smart about this and run the minimum number of calculations
        # which would be necessary for the possiblity of convergence to occur?
        # i.e. if we _know_ that one more calculation will _not_ lead to convergence,
        # we should run more than one calculation (up to convergence_window - 1)
        if not self.initialized:
            inputs = [{
                self.input_key: orm.Float(self.input_values[i])
            } for i in range(self.convergence_window)]
            self.current_index = self.convergence_window - 1
            self.initialized = True
        else:
            # Will throw an IndexError if the index is out-of-range (i.e. we've reached the end of the inputs)
            inputs = [{self.input_key: orm.Float(self.input_values[self.current_index + 1])}]
            self.current_index += 1
        return inputs

    def _update(self, outputs: ty.Dict[int, ty.Any]) -> None:
        """
        Update the state of the engine by saving all the output values and
        storing them in the result_values property
        """
        output_values = outputs.values()  # Don't need keys (AiiDA UUIDs)
        self.result_values += [val[self.result_key] for val in output_values]  # Values are AiiDA types

    def _get_optimal_result(self) -> ty.Tuple[int, Result]:
        """
        Retrieve the converged index and result value (output value, _not_ max
        distance within convergence window)
        """
        return (
            self.current_index - self.convergence_window + 1,
            self.result_values[-self.convergence_window]
        )

    @property
    def result_value(self) -> Result:
        return self._get_optimal_result()[1]

    @property
    def result_index(self) -> int:
        return self._get_optimal_result()[0]


class Convergence(OptimizationEngineWrapper):
    """
    Wrapper class for convergence engine
    """

    _IMPL_CLASS = _ConvergenceImpl

    def __new__(  # pylint: disable=too-many-arguments,dangerous-default-value,arguments-differ
        cls,
        input_values: ty.Iterable[ty.Any],
        tol: float,
        input_key: str,
        result_key: str,
        convergence_window: int = 2,
        current_index: int = 0,
        result_values: ty.List[ty.Any] = [],
        initialized: bool = False,
        logger: ty.Optional[ty.Any] = None
    ) -> _ConvergenceImpl:
        return cls._IMPL_CLASS(
            input_values=input_values,
            tol=tol,
            input_key=input_key,
            result_key=result_key,
            convergence_window=convergence_window,
            current_index=current_index,
            result_values=result_values,
            initialized=initialized,
            logger=logger
        )
