"""
Defines the optimization engine base class.
"""

from __future__ import division, print_function, unicode_literals

from abc import ABCMeta, abstractmethod, abstractproperty

import yaml
from fsc.export import export
from future.utils import with_metaclass

from ._result_mapping import ResultMapping

yaml.representer.Representer.add_representer(ABCMeta, yaml.representer.Representer.represent_name)


@export
class OptimizationEngineImpl(with_metaclass(ABCMeta, object)):
    """
    Base class for the stateful optimization engine implementation.
    """

    def __init__(self, result_state=None):
        self._result_mapping = ResultMapping.from_state(result_state)

    @classmethod
    def from_state(cls, state):
        return cls(**state)

    @property
    def state(self):
        return dict(result_state=self._result_mapping.state, **self._state)

    @abstractproperty
    def _state(self):
        pass

    @abstractproperty
    def is_finished(self):
        pass

    def create_inputs(self):
        return self._result_mapping.add_inputs(self._create_inputs())

    @abstractmethod
    def _create_inputs(self):
        pass

    def update(self, outputs):
        self._result_mapping.add_outputs(outputs)
        self._update(outputs)

    @abstractmethod
    def _update(self, outputs):
        pass

    @abstractproperty
    def result_value(self):
        pass

    @abstractproperty
    def result_index(self):
        pass


@export
class OptimizationEngineWrapper(with_metaclass(ABCMeta, object)):
    """
    Base class for wrappers that supply the public interface for optimization engines.
    """
    _IMPL_CLASS = NotImplementedError

    def __new__(cls, *args, **kwargs):
        return cls._IMPL_CLASS(*args, **kwargs)

    @classmethod
    def from_state(cls, state):
        return cls._IMPL_CLASS(**state)


@export  # pylint: disable=abstract-method
class OptimizationEngine(OptimizationEngineImpl):
    """
    Base class for stateful optimization engines.
    """

    @classmethod
    def from_state(cls, state):
        return cls(**state)
