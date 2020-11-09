# -*- coding: utf-8 -*-

# Author: Austin Zadoks <austin.zadoks@epfl.ch>
"""
Defines a basic and generic parameter convergence engine.
"""

import itertools
import typing as ty

import numpy as np
from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from .base import (OptimizationEngineImpl, OptimizationEngineWrapper)
from ._result_mapping import Result
from ..helpers import get_nested_result

__all__ = ['Convergence']


class _ConvergenceImpl(OptimizationEngineImpl):
    """
    Implementation class for the convergence engine.
    """
    def __init__(
        self,
        *,
        input_values: ty.List[ty.Any],
        tol: float,
        input_key: str,
        result_key: str,
        convergence_window: int,
        array_name: ty.Optional[str],
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
        self.array_name = array_name
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
    def _result_window(self) -> ty.List[ty.Any]:
        """
        Create a list of results corresponding to the current convergence window
        and convert those results to python / numpy objects from AiiDA objects
        """
        result_window = self.result_values[-self.convergence_window:]
        for i, result in enumerate(result_window):
            if isinstance(result, orm.ArrayData):
                result_window[i] = result.get_array(self.array_name)
            elif isinstance(result, (orm.Float, orm.Int)):
                result_window[i] = result.value

        return result_window

    @property
    def _distance_triangle(self) -> ty.List[ty.List[ty.Any]]:
        """
        Calculate the pair-wise distance between entries of a list
        into a triangle-like jagged list.
        """
        # |x_i - x_j| for i [0, N-1], j (i, N]
        distance_triangle = [[
            np.linalg.norm(self._result_window[i] - self._result_window[j])
            for j in range(i + 1, self.convergence_window)
        ] for i in range(self.convergence_window - 1)]

        return distance_triangle

    @property
    def _num_new_iters(self) -> int:
        """
        Determine the minimum number of additional outputs to have a hope
        of converging in the next step.
        """
        distance_triangle = self._distance_triangle
        # Find location of the last calculation which creates out-of-tolerance
        # roughness, and do enough calculations so that it is no longer in the
        # next convergence window
        num_new_iters = 0
        for i, row in enumerate(distance_triangle):
            if np.any(np.array(row) > self.tol):
                num_new_iters = i + 1
        # Check that we don't go past the end of the input_values when trying
        # to remove the calculation that is too rough from the window
        # If we do, return -1 as an indication that convergence will not be
        # possible
        if self.current_index + num_new_iters > len(self.input_values):
            # num_new_iters = len(self.input_values) - self.current_index
            num_new_iters = -1

        return num_new_iters

    @property
    def is_converged(self) -> bool:
        """
        Check if convergence has been reached by calculating the Frobenius
        or 2-norm of the difference between all the result values / arrays
        and checking that the maximum distance between points in the
        convergence window is less than the tolerance.
        """
        if not self.initialized:
            return False

        # calculate pair distances between results
        distance_triangle = self._distance_triangle
        # flatten all the distances into a 1D list
        distances = list(itertools.chain(*distance_triangle))

        # check if the maximum distance is less than the tolerance
        return bool(np.max(distances) < self.tol)

    @property
    def is_finished(self) -> bool:
        if not self.initialized:
            return False

        # If we've used all the input values, we're finished
        if len(self.result_values) >= len(self.input_values):
            return True

        # If _num_new_iters is -1, we know that we cannot converge with
        # the remaining inputs in input_values, so we're finished
        if self._num_new_iters == -1:
            return True

        return self.is_converged

    @property
    def is_finished_ok(self) -> bool:
        if self.is_finished and self.is_converged:
            return True
        return False

    def _create_inputs(self) -> ty.List[ty.Dict[str, orm.Float]]:
        """
        Create the inputs for the evaluation function.
        If the work chain is not initialized, the appropriate number of
        inputs to fill the convergence window are generated.
        Otherwise, one more input will be generated
        """
        if not self.initialized:
            # Do enough calculations to fill the initial convergence window
            num_new_iters = self.convergence_window
            self.initialized = True
        else:
            num_new_iters = self._num_new_iters

        self.current_index += num_new_iters
        inputs = [{
            self.input_key: to_aiida_type(self.input_values[i])
        } for i in range(self.current_index - num_new_iters, self.current_index)]

        return inputs

    def _update(self, outputs: ty.Dict[int, ty.Any]) -> None:
        """
        Update the state of the engine by saving all the output values and
        storing them in the result_values property
        """
        output_keys = sorted(outputs.keys())  # Sort keys to preserve evaluation order
        output_values = [outputs[key] for key in output_keys]
        self.result_values += [
            get_nested_result(val, self.result_key) for val in output_values
        ]  # Values are AiiDA types

    def _get_optimal_result(self) -> ty.Tuple[int, orm.Node, orm.Node]:
        """
        Retrieve the converged index and result value (output value, _not_ max
        distance within convergence window)
        """
        opt_index = len(self.result_values) - self.convergence_window
        opt_input = self._result_mapping[opt_index].input[self.input_key]
        opt_output = get_nested_result(self._result_mapping[opt_index].output, self.result_key)

        return (opt_index, opt_input, opt_output)


class Convergence(OptimizationEngineWrapper):
    """
    Wrapper class for convergence engine

    Parameters
    ----------
    input_values : iterable object
        List or other iterable of inputs within the desired range to check convergence
    tol : float
        Roughness tolerance for checking convergence
    input_key : str
        Name of the input key which should be varied to find convergence
    result_key : str
        Name of the output / result key which is the value to converge
    convergence_window : int
        Number of results to consider when checking convergence
    array_name : str or None
        Name of array within output / result ArrayData (only necessary if the output is
        given in an ArrayData)
    """

    _IMPL_CLASS = _ConvergenceImpl

    def __new__(  #type: ignore  # pylint: disable=too-many-arguments,arguments-differ
        cls,
        input_values: ty.List[ty.Any],
        tol: float,
        input_key: str,
        result_key: str,
        convergence_window: int = 2,
        array_name: ty.Optional[str] = None,
        logger: ty.Optional[ty.Any] = None,
    ) -> _ConvergenceImpl:
        return cls._IMPL_CLASS(  # pylint: disable=no-member
            input_values=input_values,
            tol=tol,
            input_key=input_key,
            result_key=result_key,
            convergence_window=convergence_window,
            array_name=array_name,
            current_index=0,
            result_values=[],
            initialized=False,
            logger=logger
        )
