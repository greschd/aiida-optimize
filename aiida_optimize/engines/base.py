# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the base classes needed to implement an optimization engine.
"""

from __future__ import annotations

import typing as ty
from abc import ABCMeta, abstractmethod

import yaml

from ._result_mapping import ResultMapping, Result

yaml.representer.Representer.add_representer(ABCMeta, yaml.representer.Representer.represent_name)

__all__ = (
    'OptimizationEngineImpl',
    'OptimizationEngineImplWithOutputs',
    'OptimizationEngineWrapper',
)


class OptimizationEngineImpl:
    """
    Base class for the stateful optimization engine implementation.
    """
    __metaclass__ = ABCMeta

    def __init__(
        self, logger: ty.Any, result_state: ty.Optional[ty.Dict[int, Result]] = None
    ) -> None:
        self._logger = logger
        self._result_mapping = ResultMapping.from_state(result_state)

    @classmethod
    def from_state(cls, state: ty.Dict[str, ty.Any]) -> OptimizationEngineImpl:
        """
        Create an instance of the class from the serialized state.
        """
        return cls(**state)

    @property
    def state(self) -> ty.Dict[str, ty.Any]:
        """
        The serialized state of the instance, including the result mapping.
        """
        return dict(result_state=self._result_mapping.state, **self._state)

    @property
    @abstractmethod
    def _state(self) -> ty.Dict[str, ty.Any]:
        """
        The serialized state of the instance, without the result mapping. This function needs to be implemented by child classes.
        """

    @property
    @abstractmethod
    def is_finished(self) -> bool:
        """
        Returns true if the optimization is finished.
        """

    @property
    def is_finished_ok(self) -> bool:
        """
        Returns true if the optimization is finished without error.
        """
        return self.is_finished

    def create_inputs(self):
        """
        Creates the inputs and adds them to the result mapping.
        """
        return self._result_mapping.add_inputs(self._create_inputs())

    @abstractmethod
    def _create_inputs(self):
        """
        Creates the inputs for evaluations that need to be launched. This function needs to be implemented by child classes.
        """

    def update(self, outputs) -> None:
        """
        Updates the result mapping and engine instance with the evaluation outputs.
        """
        self._result_mapping.add_outputs(outputs)
        self._update(outputs)

    @abstractmethod
    def _update(self, outputs: ty.Dict[int, ty.Any]) -> None:
        """
        Updates the engine instance with the evaluation outputs. This method needs to be implemented by child classes.
        """

    @property
    def result_index(self) -> int:
        """
        Returns the index (in the result mapping) of the optimal evaluation.
        """
        index, _, _ = self._get_optimal_result()
        return index

    @property
    def result_input_value(self) -> ty.Any:
        """
        Return the input value of the optimal evaluation.
        """
        _, value, _ = self._get_optimal_result()
        return value

    @property
    def result_output_value(self) -> ty.Any:
        """
        Return the output value of the optimal evaluation.
        """
        _, _, value = self._get_optimal_result()
        return value

    @abstractmethod
    def _get_optimal_result(self) -> ty.Tuple[int, ty.Any, ty.Any]:
        """
        Return the index, input value, and output value of the best evaluation process.
        """


class OptimizationEngineImplWithOutputs:
    """
    Base class for the optimization engine implementation emitting custom outputs.
    """
    @abstractmethod
    def get_engine_outputs(self) -> ty.Dict[str, ty.Any]:
        """
        Return the custom outputs created by the engine, at the end of
        the run.

        The result must be compatible with `WorkChain.out`, i.e. its
        keys are labels, and the values are either AiiDA nodes, or
        nested dictionaries.

        All AiiDA nodes returned *must* already be stored.
        """


class OptimizationEngineWrapper:
    """
    Base class for wrappers that supply the public interface for optimization engines.
    """
    __metaclass__ = ABCMeta
    _IMPL_CLASS: ty.Type[OptimizationEngineImpl]

    def __new__(cls, *args, **kwargs):
        return cls._IMPL_CLASS(*args, **kwargs)

    @classmethod
    def from_state(cls, state, logger):
        return cls._IMPL_CLASS(logger=logger, **state)
